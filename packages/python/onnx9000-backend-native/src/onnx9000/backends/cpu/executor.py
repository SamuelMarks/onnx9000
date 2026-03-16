"""CPU Execution Provider."""

import logging
from typing import Any, Dict, List
from onnx9000.backends.cpu.ops import OP_REGISTRY
from onnx9000.backends.cpu.ops_ml import ML_OP_REGISTRY
from onnx9000.core.execution import ExecutionContext, ExecutionProvider
from onnx9000.core.ir import Graph, Tensor

logger = logging.getLogger(__name__)


class CPUExecutionProvider(ExecutionProvider):
    """Executes supported nodes on the host CPU using simple fallback logic."""

    def get_supported_nodes(self, graph: Graph) -> List[str]:
        """Return node names supported by CPU."""
        supported = []
        for node in graph.nodes:
            if node.op_type in OP_REGISTRY or node.op_type in ML_OP_REGISTRY:
                supported.append(node.name or node.op_type)
        return supported

    def allocate_tensors(self, tensors: List[Tensor]) -> None:
        """Mock allocation logic for generic CPU arrays."""
        return

    def _to_numpy(self, tensor: Tensor) -> Any:
        import numpy as np
        from onnx9000.core.dtypes import DType

        if tensor.data is None:
            return np.empty(tensor.shape, dtype=np.float32)
        if isinstance(tensor.data, np.ndarray):
            return tensor.data
        dtype_mapping = {
            DType.FLOAT32: np.float32,
            DType.FLOAT64: np.float64,
            DType.INT32: np.int32,
            DType.INT64: np.int64,
            DType.INT8: np.int8,
            DType.UINT8: np.uint8,
            DType.INT16: np.int16,
            DType.UINT16: np.uint16,
            DType.BOOL: bool,
            DType.FLOAT16: np.float16,
        }
        np_dtype = dtype_mapping.get(tensor.dtype, np.float32)
        arr = np.frombuffer(tensor.data, dtype=np_dtype)
        return arr.reshape([int(x) if hasattr(x, "value") else int(x) for x in tensor.shape])

    def execute(
        self, graph: Graph, context: ExecutionContext, inputs: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Evaluate nodes sequentially."""
        import numpy as np

        results = {}
        from onnx9000.core.utils import topological_sort

        nodes = topological_sort(graph)
        for node in nodes:
            op_func = None
            if node.op_type in OP_REGISTRY:
                op_func = OP_REGISTRY[node.op_type]
            elif node.op_type in ML_OP_REGISTRY:
                op_func = ML_OP_REGISTRY[node.op_type]
            if op_func is not None:
                node_inputs = []
                for inp_name in node.inputs:
                    if inp_name in inputs:
                        node_inputs.append(self._to_numpy(inputs[inp_name]))
                    elif inp_name in results:
                        node_inputs.append(self._to_numpy(results[inp_name]))
                    else:
                        node_inputs.append(np.zeros((1,)))
                out_tensors = op_func(
                    node_inputs, {k: v.value for (k, v) in node.attributes.items()}
                )
                for i, out_name in enumerate(node.outputs):
                    if i < len(out_tensors):
                        t = Tensor(
                            out_name,
                            tuple(out_tensors[i].shape),
                            inputs[list(inputs.keys())[0]].dtype if inputs else None,
                            data=out_tensors[i],
                        )
                        results[out_name] = t
        return results
