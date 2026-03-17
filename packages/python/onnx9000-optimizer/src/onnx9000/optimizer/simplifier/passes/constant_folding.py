"""Provides constant_folding.py module functionality."""

import logging

import numpy as np
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.simplifier.passes.base import GraphPass

logger = logging.getLogger(__name__)
MAX_CONSTANT_SIZE_BYTES = 10 * 1024 * 1024


class ConstantFoldingPass(GraphPass):
    """
    Evaluates deterministic operations at compile-time when all of their
    inputs are statically known constants or initializers.
    """

    def run(self, graph: Graph) -> bool:
        """Implements the run method or operation."""
        changed = False
        while True:
            iteration_changed = self._run_once(graph)
            if not iteration_changed:
                break
            changed = True
        return changed

    def _run_once(self, graph: Graph) -> bool:
        """Implements the _run_once method or operation."""
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
            if node.op_type == "Constant":
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
                    result = self._evaluate_node(node.op_type, input_vals, node.attributes, node)
                    if result is not None:
                        if isinstance(result, list):
                            size = sum(r.nbytes for r in result)
                        else:
                            size = result.nbytes
                        if size <= MAX_CONSTANT_SIZE_BYTES:
                            if isinstance(result, list):
                                if len(result) > len(node.outputs):
                                    raise IndexError("Split result length exceeds node outputs")
                                for idx, res in enumerate(result):
                                    const_node = Node(
                                        op_type="Constant",
                                        inputs=[],
                                        outputs=[node.outputs[idx]],
                                        attributes={"value": res},
                                        name=f"folded_{node.name}_{idx}",
                                    )
                                    new_nodes.append(const_node)
                                    known_values[node.outputs[idx]] = res
                            else:
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
                (new_node, local_changed) = self._partial_fold(node, known_values)
                if local_changed:
                    changed = True
                    if new_node is not None:
                        new_nodes.append(new_node)
                else:
                    new_nodes.append(node)
        graph.nodes = new_nodes
        return changed

    def _evaluate_node(self, op_type: str, inputs: list, attrs: dict, node=None):
        """Implements the _evaluate_node method or operation."""
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
        if op_type == "Sin":
            return np.sin(inputs[0])
        if op_type == "Cos":
            return np.cos(inputs[0])
        if op_type == "Tan":
            return np.tan(inputs[0])
        if op_type == "Asin":
            return np.arcsin(inputs[0])
        if op_type == "Acos":
            return np.arccos(inputs[0])
        if op_type == "Atan":
            return np.arctan(inputs[0])
        if op_type == "Sinh":
            return np.sinh(inputs[0])
        if op_type == "Cosh":
            return np.cosh(inputs[0])
        if op_type == "Tanh":
            return np.tanh(inputs[0])
        if op_type == "Neg":
            return np.negative(inputs[0])
        if op_type == "Sign":
            return np.sign(inputs[0])
        if op_type == "Ceil":
            return np.ceil(inputs[0])
        if op_type == "Floor":
            return np.floor(inputs[0])
        if op_type == "Round":
            return np.round(inputs[0])
        if op_type == "Mod":
            return np.mod(inputs[0], inputs[1])
        if op_type == "Max":
            return np.maximum.reduce(inputs) if len(inputs) > 1 else inputs[0]
        if op_type == "Min":
            return np.minimum.reduce(inputs) if len(inputs) > 1 else inputs[0]
        if op_type == "Clip":
            min_val = inputs[1] if len(inputs) > 1 and inputs[1] is not None else None
            max_val = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
            return np.clip(inputs[0], min_val, max_val)
        if op_type == "And":
            return np.logical_and(inputs[0], inputs[1])
        if op_type == "Or":
            return np.logical_or(inputs[0], inputs[1])
        if op_type == "Not":
            return np.logical_not(inputs[0])
        if op_type == "Xor":
            return np.logical_xor(inputs[0], inputs[1])
        if op_type == "Equal":
            return np.equal(inputs[0], inputs[1])
        if op_type == "Greater":
            return np.greater(inputs[0], inputs[1])
        if op_type == "Less":
            return np.less(inputs[0], inputs[1])
        if op_type == "GreaterOrEqual":
            return np.greater_equal(inputs[0], inputs[1])
        if op_type == "LessOrEqual":
            return np.less_equal(inputs[0], inputs[1])
        if op_type == "BitShift":
            direction = attrs.get("direction", "")
            if direction == "RIGHT":
                return np.right_shift(inputs[0], inputs[1])
            elif direction == "LEFT":
                return np.left_shift(inputs[0], inputs[1])
            return None
        if op_type == "BitwiseAnd":
            return np.bitwise_and(inputs[0], inputs[1])
        if op_type == "BitwiseOr":
            return np.bitwise_or(inputs[0], inputs[1])
        if op_type == "BitwiseNot":
            return np.bitwise_not(inputs[0])
        if op_type == "BitwiseXor":
            return np.bitwise_xor(inputs[0], inputs[1])
        if op_type == "IsInf":
            detect_positive = attrs.get("detect_positive", 1)
            detect_negative = attrs.get("detect_negative", 1)
            if detect_positive and detect_negative:
                return np.isinf(inputs[0])
            elif detect_positive:
                return np.isposinf(inputs[0])
            elif detect_negative:
                return np.isneginf(inputs[0])
            return np.zeros_like(inputs[0], dtype=bool)
        if op_type == "IsNaN":
            return np.isnan(inputs[0])
        if op_type == "Erf":
            from scipy.special import erf

            return erf(inputs[0])
        if op_type == "Relu":
            return np.maximum(inputs[0], 0)
        if op_type == "Sigmoid":
            return 1 / (1 + np.exp(-inputs[0]))
        if op_type == "ReduceSum":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.sum(inputs[0], axis=axes, keepdims=bool(keepdims))
        if op_type == "ReduceMean":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.mean(inputs[0], axis=axes, keepdims=bool(keepdims))
        if op_type == "ReduceMax":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.max(inputs[0], axis=axes, keepdims=bool(keepdims))
        if op_type == "ReduceMin":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.min(inputs[0], axis=axes, keepdims=bool(keepdims))
        if op_type == "ReduceProd":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.prod(inputs[0], axis=axes, keepdims=bool(keepdims))
        if op_type == "ReduceL1":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.sum(np.abs(inputs[0]), axis=axes, keepdims=bool(keepdims))
        if op_type == "ReduceL2":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.sqrt(np.sum(np.square(inputs[0]), axis=axes, keepdims=bool(keepdims)))
        if op_type == "ReduceLogSum":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            return np.log(np.sum(inputs[0], axis=axes, keepdims=bool(keepdims)))
        if op_type == "ReduceLogSumExp":
            axes = attrs.get("axes")
            keepdims = attrs.get("keepdims", 1)
            if axes is not None:
                axes = tuple(axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes = tuple(inputs[1].tolist())
            m = np.max(inputs[0], axis=axes, keepdims=True)
            res = m + np.log(np.sum(np.exp(inputs[0] - m), axis=axes, keepdims=True))
            if not keepdims:
                res = np.squeeze(res, axis=axes)
            return res
        if op_type == "CastLike":
            return inputs[0].astype(inputs[1].dtype)
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
            (data, starts, ends) = (inputs[0], inputs[1], inputs[2])
            axes = (
                inputs[3] if len(inputs) > 3 and inputs[3] is not None else list(range(len(starts)))
            )
            steps = inputs[4] if len(inputs) > 4 and inputs[4] is not None else [1] * len(starts)
            slices = [slice(None)] * data.ndim
            for i, ax in enumerate(axes):
                slices[ax] = slice(starts[i], ends[i], steps[i])
            return data[tuple(slices)]
        if op_type == "Split":
            axis = attrs.get("axis", 0)
            if len(inputs) > 1 and inputs[1] is not None:
                split = inputs[1].tolist()
                indices = np.cumsum(split)[:-1]
                return np.split(inputs[0], indices, axis=axis)
            else:
                num_outputs = len(node.outputs)
                return np.split(inputs[0], num_outputs, axis=axis)
        if op_type == "Expand":
            return np.broadcast_to(inputs[0], tuple(inputs[1]))
        if op_type == "Pad":
            pads = inputs[1]
            mode = attrs.get("mode", "constant")
            value = inputs[2] if len(inputs) > 2 and inputs[2] is not None else 0
            if isinstance(value, np.ndarray):
                value = value.item() if value.size == 1 else value[0]
            ndims = inputs[0].ndim
            pad_width = [(pads[i], pads[i + ndims]) for i in range(ndims)]
            if mode == "constant":
                return np.pad(inputs[0], pad_width, mode=mode, constant_values=value)
            elif mode == "reflect":
                return np.pad(inputs[0], pad_width, mode="reflect")
            elif mode == "edge":
                return np.pad(inputs[0], pad_width, mode="edge")
            return None
        if op_type == "ConstantOfShape":
            shape = tuple(inputs[0].tolist()) if len(inputs) > 0 else ()
            val = attrs.get("value", np.array([0], dtype=np.float32))
            if isinstance(val, Tensor):
                val = val.data
            elif isinstance(val, list):
                val = np.array(val)
            val = val.item() if val.size == 1 else val[0]
            return np.full(
                shape, val, dtype=type(val) if not isinstance(val, float) else np.float32
            )
        if op_type == "Where":
            return np.where(inputs[0], inputs[1], inputs[2])
        if op_type == "CumSum":
            axis = inputs[1].item() if inputs[1].size == 1 else inputs[1][0]
            exclusive = attrs.get("exclusive", 0)
            reverse = attrs.get("reverse", 0)
            data = inputs[0]
            if reverse:
                data = np.flip(data, axis=axis)
            res = np.cumsum(data, axis=axis)
            if exclusive:
                shape = list(data.shape)
                shape[axis] = 1
                zeros = np.zeros(shape, dtype=data.dtype)
                res = np.concatenate(
                    (zeros, res.take(indices=range(res.shape[axis] - 1), axis=axis)), axis=axis
                )
            if reverse:
                res = np.flip(res, axis=axis)
            return res
        if op_type == "Trilu":
            k = inputs[1].item() if len(inputs) > 1 and inputs[1] is not None else 0
            upper = attrs.get("upper", 1)
            if upper:
                return np.triu(inputs[0], k=k)
            else:
                return np.tril(inputs[0], k=k)
        if op_type in ("MatMul", "Gemm"):
            if op_type == "MatMul":
                return np.matmul(inputs[0], inputs[1])
            else:
                transA = attrs.get("transA", 0)
                transB = attrs.get("transB", 0)
                alpha = attrs.get("alpha", 1.0)
                beta = attrs.get("beta", 1.0)
                A = inputs[0].T if transA else inputs[0]
                B = inputs[1].T if transB else inputs[1]
                res = alpha * np.matmul(A, B)
                if len(inputs) > 2 and inputs[2] is not None:
                    res += beta * inputs[2]
                return res
        if op_type == "Conv":
            return None
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
        """Implements the _partial_fold method or operation."""
        if node.op_type == "Add":
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 0):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[0]])
                return (node, True)
            if node.inputs[0] in known_values and np.all(known_values[node.inputs[0]] == 0):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[1]])
                return (node, True)
        if node.op_type == "Mul":
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 1):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[0]])
                return (node, True)
            if node.inputs[0] in known_values and np.all(known_values[node.inputs[0]] == 1):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[1]])
                return (node, True)
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 0):
                return (
                    Node(
                        "Constant",
                        [],
                        node.outputs,
                        {"value": np.zeros_like(known_values[node.inputs[1]])},
                        f"partial_{node.name}",
                    ),
                    True,
                )
        if node.op_type == "Sub":
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 0):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[0]])
                return (node, True)
            if node.inputs[0] == node.inputs[1]:
                return (
                    Node(
                        "Constant",
                        [],
                        node.outputs,
                        {"value": np.zeros(1, dtype=np.float32)},
                        f"partial_{node.name}",
                    ),
                    True,
                )
        if node.op_type == "Where" and node.inputs[0] in known_values:
            cond = known_values[node.inputs[0]]
            if isinstance(cond, np.ndarray) and (cond.size == 1 or np.all(cond == cond.flat[0])):
                if np.all(cond):
                    (node.op_type, node.inputs) = ("Identity", [node.inputs[1]])
                else:
                    (node.op_type, node.inputs) = ("Identity", [node.inputs[2]])
                return (node, True)
        if node.op_type == "Div":
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 1):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[0]])
                return (node, True)
            if node.inputs[0] == node.inputs[1]:
                return (
                    Node(
                        "Constant",
                        [],
                        node.outputs,
                        {"value": np.ones(1, dtype=np.float32)},
                        f"partial_{node.name}",
                    ),
                    True,
                )
        if node.op_type == "Pow":
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 1):
                (node.op_type, node.inputs) = ("Identity", [node.inputs[0]])
                return (node, True)
            if node.inputs[1] in known_values and np.all(known_values[node.inputs[1]] == 0):
                return (
                    Node(
                        "Constant",
                        [],
                        node.outputs,
                        {"value": np.ones(1, dtype=np.float32)},
                        f"partial_{node.name}",
                    ),
                    True,
                )
        return (node, False)


def constant_folding(graph: Graph) -> None:
    """Implements the constant_folding method or operation."""
    ConstantFoldingPass().run(graph)
