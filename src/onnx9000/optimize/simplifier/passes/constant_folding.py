"""Provides constant_folding.py module functionality."""

import logging
import numpy as np

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimize.simplifier.passes.base import GraphPass

logger = logging.getLogger(__name__)

MAX_CONSTANT_SIZE_BYTES = 10 * 1024 * 1024


class ConstantFoldingPass(GraphPass):
    """
    Evaluates deterministic operations at compile-time when all of their
    inputs are statically known constants or initializers.
    """

    def run(self, graph: Graph) -> bool:
        """Provides run functionality and verification."""
        changed = False
        while True:
            iteration_changed = self._run_once(graph)
            if not iteration_changed:
                break
            changed = True
        return changed

    def _run_once(self, graph: Graph) -> bool:
        """Provides  run once functionality and verification."""
        changed = False
        known_values = {}

        for init_name in graph.initializers:
            if init_name in graph.tensors and graph.tensors[init_name].data is not None:
                known_values[init_name] = graph.tensors[init_name].data

        for node in graph.nodes:
            if node.op_type == "Constant":
                val = node.attributes.get("value")
                if isinstance(val, Tensor) and val.data is not None:
                    known_values[node.outputs[0]] = val.data
                elif isinstance(val, np.ndarray):
                    known_values[node.outputs[0]] = val
                elif "value_float" in node.attributes:
                    known_values[node.outputs[0]] = np.array(
                        node.attributes["value_float"], dtype=np.float32
                    )
                elif "value_int" in node.attributes:
                    known_values[node.outputs[0]] = np.array(
                        node.attributes["value_int"], dtype=np.int64
                    )

        new_nodes = []
        for node in graph.nodes:
            if node.op_type in ("Constant", "ConstantOfShape"):
                new_nodes.append(node)
                continue

            all_known = True
            input_vals = []
            for inp in node.inputs:
                if inp in known_values:
                    input_vals.append(known_values[inp])
                elif inp == "":
                    input_vals.append(None)
                else:
                    all_known = False
                    break

            folded = False
            if all_known and node.op_type not in ("RandomUniform", "RandomNormal"):
                try:
                    result = self._evaluate_node(
                        node.op_type, input_vals, node.attributes
                    )
                    if result is not None:
                        if result.nbytes <= MAX_CONSTANT_SIZE_BYTES:
                            const_node = Node(
                                op_type="Constant",
                                inputs=[],
                                outputs=list(node.outputs),
                                attributes={"value": result},
                                name=f"folded_{node.name}",
                            )
                            new_nodes.append(const_node)
                            known_values[node.outputs[0]] = result
                            logger.info(f"Folded node {node.name} ({node.op_type})")
                            folded = True
                            changed = True
                except Exception as e:
                    logger.debug(f"Failed to fold {node.name}: {e}")

            if not folded:
                new_node, local_changed = self._partial_fold(node, known_values)
                if local_changed:
                    changed = True
                    if new_node is not None:
                        new_nodes.append(new_node)
                else:
                    new_nodes.append(node)

        graph.nodes = new_nodes
        return changed

    def _evaluate_node(self, op_type: str, inputs: list, attrs: dict):
        """Provides  evaluate node functionality and verification."""
        if op_type == "Add":
            return np.add(inputs[0], inputs[1])
        if op_type == "Sub":
            return np.subtract(inputs[0], inputs[1])
        if op_type == "Mul":
            return np.multiply(inputs[0], inputs[1])
        if op_type == "Div":
            return np.divide(inputs[0], inputs[1])
        if op_type == "Pow":
            return np.power(inputs[0], inputs[1])
        if op_type == "Abs":
            return np.abs(inputs[0])
        if op_type == "Exp":
            return np.exp(inputs[0])
        if op_type == "Log":
            return np.log(inputs[0])
        if op_type == "Sqrt":
            return np.sqrt(inputs[0])
        if op_type == "Cast":
            to_type = attrs.get("to")
            dtype_map = {
                1: np.float32,
                2: np.uint8,
                3: np.int8,
                4: np.uint16,
                5: np.int16,
                6: np.int32,
                7: np.int64,
                10: np.float16,
                11: np.float64,
                9: bool,
            }
            if to_type in dtype_map:
                return inputs[0].astype(dtype_map[to_type])
            return inputs[0]
        if op_type == "Reshape":
            return inputs[0].reshape(inputs[1])
        if op_type == "Transpose":
            return np.transpose(inputs[0], axes=attrs.get("perm"))
        if op_type == "Squeeze":
            axes = attrs.get("axes")
            if len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.squeeze(inputs[0], axis=tuple(axes) if axes is not None else None)
        if op_type == "Unsqueeze":
            axes = attrs.get("axes")
            if len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            res = inputs[0]
            for ax in sorted(axes):
                res = np.expand_dims(res, axis=ax)
            return res
        if op_type == "Flatten":
            axis = attrs.get("axis", 1)
            return inputs[0].reshape((np.prod(inputs[0].shape[:axis], dtype=int), -1))
        if op_type == "Concat":
            return np.concatenate(inputs, axis=attrs.get("axis", 0))
        if op_type == "Slice":
            data, starts, ends = inputs[0], inputs[1], inputs[2]
            axes = (
                inputs[3]
                if len(inputs) > 3 and inputs[3] is not None
                else list(range(len(starts)))
            )
            steps = (
                inputs[4]
                if len(inputs) > 4 and inputs[4] is not None
                else [1] * len(starts)
            )
            slices = [slice(None)] * data.ndim
            for i, ax in enumerate(axes):
                slices[ax] = slice(starts[i], ends[i], steps[i])
            return data[tuple(slices)]
        if op_type == "Gather":
            return np.take(inputs[0], inputs[1], axis=attrs.get("axis", 0))
        if op_type == "Shape":
            return np.array(inputs[0].shape, dtype=np.int64)
        if op_type == "Size":
            return np.array(inputs[0].size, dtype=np.int64)
        if op_type == "NonZero":
            return np.array(np.nonzero(inputs[0]), dtype=np.int64)
        return None

    def _partial_fold(self, node: Node, known_values: dict) -> tuple:
        """Provides  partial fold functionality and verification."""
        if node.op_type == "Add":
            if node.inputs[1] in known_values and np.all(
                known_values[node.inputs[1]] == 0
            ):
                node.op_type, node.inputs = "Identity", [node.inputs[0]]
                return node, True
            if node.inputs[0] in known_values and np.all(
                known_values[node.inputs[0]] == 0
            ):
                node.op_type, node.inputs = "Identity", [node.inputs[1]]
                return node, True
        if node.op_type == "Mul":
            if node.inputs[1] in known_values and np.all(
                known_values[node.inputs[1]] == 1
            ):
                node.op_type, node.inputs = "Identity", [node.inputs[0]]
                return node, True
            if node.inputs[0] in known_values and np.all(
                known_values[node.inputs[0]] == 1
            ):
                node.op_type, node.inputs = "Identity", [node.inputs[1]]
                return node, True
            if node.inputs[1] in known_values and np.all(
                known_values[node.inputs[1]] == 0
            ):
                return Node(
                    "Constant",
                    [],
                    node.outputs,
                    {"value": np.zeros_like(known_values[node.inputs[1]])},
                    f"partial_{node.name}",
                ), True
        if node.op_type == "Pow":
            if node.inputs[1] in known_values and np.all(
                known_values[node.inputs[1]] == 1
            ):
                node.op_type, node.inputs = "Identity", [node.inputs[0]]
                return node, True
        if node.op_type == "Div":
            if node.inputs[1] in known_values and np.all(
                known_values[node.inputs[1]] == 1
            ):
                node.op_type, node.inputs = "Identity", [node.inputs[0]]
                return node, True
        return node, False


def constant_folding(graph: Graph) -> None:
    """Provides constant folding functionality and verification."""
    ConstantFoldingPass().run(graph)
