"""InferenceSession orchestration logic routing Graph nodes to Execution Providers."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.core.execution import ExecutionContext, ExecutionProvider, RunOptions, SessionOptions
from onnx9000.core.ir import Graph, Tensor
from onnx9000.core.utils import topological_sort

logger = logging.getLogger(__name__)


class InferenceSessionError(Onnx9000Error):
    """Raised when the InferenceSession fails to execute a graph."""


class NodeArg:
    """Abstraction for inputs/outputs properties."""

    def __init__(self, name: str, shape: tuple[Any, ...], dtype: str) -> None:
        """Initialize the NodeArg with standard properties."""
        self.name = name
        self.shape = shape
        self.type = dtype


class IOBinding:
    """Class for pre-allocated memory handling."""

    def __init__(self, session: "InferenceSession") -> None:
        """Initialize IOBinding."""
        self.session = session
        self.inputs: dict[str, Tensor] = {}
        self.outputs: dict[str, Tensor] = {}

    def bind_input(
        self,
        name: str,
        device_type: str,
        device_id: int,
        element_type: str,
        shape: tuple[int, ...],
        buffer_ptr: int,
    ) -> None:
        """Bind an input."""
        self.inputs[name] = Tensor(name, shape, DType[element_type.upper()])

    def bind_output(
        self,
        name: str,
        device_type: str,
        device_id: int,
        element_type: str,
        shape: tuple[int, ...],
        buffer_ptr: int,
    ) -> None:
        """Bind an output."""
        self.outputs[name] = Tensor(name, shape, DType[element_type.upper()])

    def bind_ortvalue_input(self, name: str, ortvalue: Tensor) -> None:
        """Bind an OrtValue input."""
        self.inputs[name] = ortvalue

    def bind_ortvalue_output(self, name: str, ortvalue: Tensor) -> None:
        """Bind an OrtValue output."""
        self.outputs[name] = ortvalue

    def synchronize_inputs(self) -> None:
        """Synchronize inputs."""
        return

    def synchronize_outputs(self) -> None:
        """Synchronize outputs."""
        return


class InferenceSession:
    """Coordinates execution of an ONNX Graph across multiple Execution Providers."""

    def __init__(
        self,
        graph: Graph,
        providers: Optional[list[ExecutionProvider]] = None,
        options: Optional[SessionOptions] = None,
    ) -> None:
        """Initialize the InferenceSession with the target graph and providers."""
        if not isinstance(graph, Graph):
            raise TypeError(
                "InferenceSession requires an IR Graph object. Parsers are decoupled from execution."
            )
        self.graph = graph
        self.providers = providers or []
        self.options = options or SessionOptions()
        self._partition_graph()

    def _partition_graph(self) -> None:
        """Assign each node in the graph to the most preferred Execution Provider.

        that supports it. Providers are ordered by preference (e.g. CUDA, then CPU).
        """
        self.node_to_provider: dict[str, ExecutionProvider] = {}
        if not self.providers:
            return
        unassigned_nodes = []
        tensor_provider: dict[str, ExecutionProvider] = {}
        for node in self.graph.nodes:
            assigned = False
            for provider in self.providers:
                if node.op_type in provider.get_supported_nodes(self.graph):
                    self.node_to_provider[node.name or node.op_type] = provider
                    assigned = True
                    break
            if not assigned:
                unassigned_nodes.append(node.name or node.op_type)
            else:
                provider = self.node_to_provider[node.name or node.op_type]
                for out in node.outputs:
                    tensor_provider[out] = provider
        if unassigned_nodes:
            raise InferenceSessionError(f"No Execution Provider supports nodes: {unassigned_nodes}")
        new_nodes = []
        for node in self.graph.nodes:
            provider = self.node_to_provider[node.name or node.op_type]
            new_inputs = []
            for inp in node.inputs:
                if inp in tensor_provider and tensor_provider[inp] is not provider:
                    copy_node_name = f"Memcpy_{inp}_to_{provider.__class__.__name__}"
                    new_inp_name = f"{inp}_copied_for_{provider.__class__.__name__}"
                    op_type = (
                        "MemcpyToDevice"
                        if "CPU" not in provider.__class__.__name__
                        else "MemcpyToHost"
                    )
                    from onnx9000.core.ir import Node

                    copy_node = Node(
                        op_type,
                        inputs=[inp],
                        outputs=[new_inp_name],
                        attributes={},
                        name=copy_node_name,
                    )
                    new_nodes.append(copy_node)
                    self.node_to_provider[copy_node_name] = provider
                    tensor_provider[new_inp_name] = provider
                    new_inputs.append(new_inp_name)
                else:
                    new_inputs.append(inp)
            node.inputs = new_inputs
            new_nodes.append(node)
        self.graph.nodes = new_nodes

    def get_inputs(self) -> list[NodeArg]:
        """Return NodeArg abstractions for inputs."""
        return [NodeArg(inp.name, inp.shape, inp.dtype.name) for inp in self.graph.inputs]

    def get_outputs(self) -> list[NodeArg]:
        """Return NodeArg abstractions for outputs."""
        return [NodeArg(out.name, out.shape, out.dtype.name) for out in self.graph.outputs]

    def get_overridable_initializers(self) -> list[NodeArg]:
        """Return NodeArg abstractions for overridable initializers."""
        res = []
        for name in self.graph.initializers:
            if name in self.graph.tensors:
                t = self.graph.tensors[name]
                res.append(NodeArg(t.name, t.shape, t.dtype.name))
        return res

    def get_providers(self) -> list[str]:
        """Return names of execution providers."""
        return [p.__class__.__name__ for p in self.providers]

    def get_provider_options(self) -> dict[str, dict[str, Any]]:
        """Return options for execution providers."""
        return {p.__class__.__name__: p.options for p in self.providers}

    def set_providers(self, providers: list[ExecutionProvider]) -> None:
        """Set execution providers dynamically."""
        self.providers = providers
        self._partition_graph()

    def run(
        self,
        output_names: Optional[list[str]],
        input_feed: dict[str, Tensor],
        run_options: Optional[RunOptions] = None,
    ) -> list[Tensor]:
        """Execute the graph and return the requested outputs."""
        if not output_names:
            output_names = [v.name for v in self.graph.outputs]
        context = ExecutionContext(options=self.options)
        current_tensors: dict[str, Tensor] = {**input_feed}
        for init_name in self.graph.initializers:
            if init_name in self.graph.tensors:
                current_tensors[init_name] = self.graph.tensors[init_name]
        sorted_nodes = topological_sort(self.graph)
        for node in sorted_nodes:
            if not self.providers:
                continue
            provider = self.node_to_provider[node.name or node.op_type]
            node_inputs = {
                inp: current_tensors[inp] for inp in node.inputs if inp in current_tensors
            }
            outputs = provider.execute(self.graph, context, node_inputs)
            current_tensors.update(outputs)
        results: list[Tensor] = []
        for out in output_names:
            if out not in current_tensors:
                raise InferenceSessionError(f"Requested output {out} was not computed.")
            results.append(current_tensors[out])
        return results

    def run_with_iobinding(
        self, iobinding: IOBinding, run_options: Optional[RunOptions] = None
    ) -> None:
        """Run with IOBinding."""
        outputs = self.run(list(iobinding.outputs.keys()), iobinding.inputs, run_options)
        for key, tensor in zip(iobinding.outputs.keys(), outputs):
            iobinding.outputs[key] = tensor
