"""Frontend Sub-Package.

Provides tracing and PyTorch-like interfaces to define and capture
computation graphs from native Python execution.
"""

from typing import Any, Union

from onnx9000.core.dtypes import DType


def infer_elementwise_shape(
    shape1: tuple[Union[int, str]], shape2: tuple[Union[int, str]]
) -> tuple[Union[int, str]]:
    """Basic NumPy-style broadcasting shape inference."""
    (len1, len2) = (len(shape1), len(shape2))
    max_len = max(len1, len2)
    s1 = (1,) * (max_len - len1) + shape1
    s2 = (1,) * (max_len - len2) + shape2
    out_shape = []
    for d1, d2 in zip(s1, s2):
        if d1 == 1:
            out_shape.append(d2)
        elif d2 == 1 or d1 == d2:
            out_shape.append(d1)
        elif isinstance(d1, str) or isinstance(d2, str):
            out_shape.append(d2 if d1 == 1 else d1)
        else:
            raise ValueError(f"Incompatible shapes for broadcasting: {shape1} and {shape2}")
    return tuple(out_shape)


def infer_matmul_shape(
    shape1: tuple[Union[int, str]], shape2: tuple[Union[int, str]]
) -> tuple[Union[int, str]]:
    """Basic MatMul shape inference."""
    if len(shape1) < 1 or len(shape2) < 1:
        raise ValueError("MatMul requires arrays of rank >= 1.")
    if len(shape1) == 1 and len(shape2) == 1:
        return ()
    if len(shape1) == 1:
        return shape2[1:]
    if len(shape2) == 1:
        return shape1[:-1]
    batch_shape = infer_elementwise_shape(shape1[:-2], shape2[:-2])
    return batch_shape + (shape1[-2], shape2[-1])


def record_op(op_type: str, inputs: list[Any], attributes: dict = None) -> Any:
    """Helper to record an operation if a Tracing context is active."""
    import numpy as np
    from onnx9000.converters.frontend.builder import get_active_builder
    from onnx9000.converters.frontend.tensor import Node, Parameter, Tensor

    builder = get_active_builder()
    if builder is None:
        raise RuntimeError(f"Cannot execute {op_type} outside of a onnx9000.Tracing context.")
    processed_inputs = []
    for inp in inputs:
        if isinstance(inp, Tensor):
            if isinstance(inp, Parameter):
                if not any(inp is p for p in builder.parameters):
                    builder.parameters.append(inp)
            processed_inputs.append(inp)
        else:
            if not isinstance(inp, np.ndarray):
                inp = np.asarray(inp)
            dt = DType.FLOAT32
            if inp.dtype == np.int64:
                dt = DType.INT64
            elif inp.dtype == np.int32:
                dt = DType.INT32
            elif inp.dtype == np.float64:
                dt = DType.FLOAT64
            elif inp.dtype == bool:
                dt = DType.BOOL
            builder._tensor_counter += 1
            name = f"constant_{builder._tensor_counter}"
            t = Parameter(name=name, shape=inp.shape, dtype=dt, data=inp)
            builder.parameters.append(t)
            processed_inputs.append(t)
    inputs = processed_inputs
    out_shape = inputs[0].shape if inputs else (1,)
    out_dtype = inputs[0].dtype if inputs else DType.FLOAT32
    if op_type in ["Trilu"]:
        out_dtype = inputs[0].dtype if inputs else DType.FLOAT32
    if op_type in ["TopK", "NonMaxSuppression"] and op_type == "TopK":
        outputs = [
            Tensor(shape=out_shape, dtype=out_dtype),
            Tensor(shape=out_shape, dtype=DType.INT64),
        ]
        node = Node(
            op_type=op_type,
            inputs=[inp.name for inp in inputs],
            outputs=[out.name for out in outputs],
            attributes=attributes or {},
        )
        builder.add_node(node)
        return outputs
    if op_type in ["DynamicQuantizeLinear"]:
        outputs = [
            Tensor(shape=out_shape, dtype=DType.UINT8),
            Tensor(shape=(), dtype=DType.FLOAT32),
            Tensor(shape=(), dtype=DType.UINT8),
        ]
        node = Node(
            op_type=op_type,
            inputs=[inp.name for inp in inputs],
            outputs=[out.name for out in outputs],
            attributes=attributes or {},
        )
        builder.add_node(node)
        return outputs
    if op_type in ["Dropout"]:
        outputs = [Tensor(shape=out_shape, dtype=out_dtype) for _ in range(2)]
        node = Node(
            op_type=op_type,
            inputs=[inp.name for inp in inputs],
            outputs=[out.name for out in outputs],
            attributes=attributes or {},
        )
        builder.add_node(node)
        return outputs
    if op_type in ["Unique"]:
        outputs = [Tensor(shape=out_shape, dtype=out_dtype) for _ in range(4)]
        node = Node(
            op_type=op_type,
            inputs=[inp.name for inp in inputs],
            outputs=[out.name for out in outputs],
            attributes=attributes or {},
        )
        builder.add_node(node)
        return outputs
    if op_type in ("Add", "Sub", "Mul", "Div"):
        if len(inputs) == 2:
            out_shape = infer_elementwise_shape(inputs[0].shape, inputs[1].shape)
    elif op_type == "MatMul":
        if len(inputs) == 2:
            out_shape = infer_matmul_shape(inputs[0].shape, inputs[1].shape)
    elif op_type == "Transpose":
        perm = attributes.get("perm") if attributes else None
        if perm:
            out_shape = tuple(inputs[0].shape[i] for i in perm)
        else:
            out_shape = tuple(reversed(inputs[0].shape))
    output_tensor = Tensor(shape=out_shape, dtype=out_dtype)
    node = Node(
        op_type=op_type,
        inputs=inputs,
        outputs=[output_tensor],
        attributes=attributes,
        name=f"{op_type}_{output_tensor.name}",
    )
    builder.add_node(node)
    return output_tensor
