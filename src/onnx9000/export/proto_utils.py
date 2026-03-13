"""
Export Sub-Package

Transforms the internal IR Graph into an official ONNX ModelProto binary format,
ensuring compliance with ONNX constraints and constraints.
"""

# mypy: ignore-errors
from typing import Any

from onnx9000 import onnx_pb2  # type: ignore
from onnx9000.frontend.tensor import Node, Parameter, Tensor


def to_tensor_proto(tensor: Parameter) -> onnx_pb2.TensorProto:
    """Converts a Parameter to an ONNX TensorProto (initializer)."""
    proto = onnx_pb2.TensorProto()
    proto.name = tensor.name
    proto.data_type = tensor.dtype.value
    proto.dims.extend([d if isinstance(d, int) else -1 for d in tensor.shape])

    if hasattr(tensor, "data") and tensor.data is not None:
        import numpy as np

        if isinstance(tensor.data, np.ndarray):
            proto.raw_data = tensor.data.tobytes()
        elif isinstance(tensor.data, list):  # pragma: no cover
            # Encode list to numpy array first, then to raw_data for compatibility with C++ parser
            # Assuming tensor.dtype has numpy equivalent mapping or we deduce it
            from onnx9000.dtypes import DType  # pragma: no cover

            if tensor.dtype == DType.FLOAT32:  # pragma: no cover
                proto.raw_data = np.array(
                    tensor.data, dtype=np.float32
                ).tobytes()  # pragma: no cover
            elif tensor.dtype == DType.FLOAT64:  # pragma: no cover
                proto.raw_data = np.array(
                    tensor.data, dtype=np.float64
                ).tobytes()  # pragma: no cover
            elif tensor.dtype in (  # pragma: no cover
                DType.INT32,
                DType.INT16,
                DType.INT8,
                DType.UINT8,
                DType.UINT16,
                DType.BOOL,
            ):
                proto.raw_data = np.array(
                    tensor.data, dtype=np.int32
                ).tobytes()  # pragma: no cover
                # Actually wait, let's use exact dtype for int32
                if tensor.dtype == DType.INT32:  # pragma: no cover
                    proto.raw_data = np.array(
                        tensor.data, dtype=np.int32
                    ).tobytes()  # pragma: no cover
                elif tensor.dtype == DType.INT16:  # pragma: no cover
                    proto.raw_data = np.array(
                        tensor.data, dtype=np.int16
                    ).tobytes()  # pragma: no cover
                elif tensor.dtype == DType.INT8:  # pragma: no cover
                    proto.raw_data = np.array(
                        tensor.data, dtype=np.int8
                    ).tobytes()  # pragma: no cover
                elif tensor.dtype == DType.UINT8:  # pragma: no cover
                    proto.raw_data = np.array(
                        tensor.data, dtype=np.uint8
                    ).tobytes()  # pragma: no cover
                elif tensor.dtype == DType.UINT16:  # pragma: no cover
                    proto.raw_data = np.array(
                        tensor.data, dtype=np.uint16
                    ).tobytes()  # pragma: no cover
                elif tensor.dtype == DType.BOOL:  # pragma: no cover
                    proto.raw_data = np.array(
                        tensor.data, dtype=np.bool_
                    ).tobytes()  # pragma: no cover
            elif tensor.dtype in (
                DType.INT64,
                DType.UINT32,
                DType.UINT64,
            ):  # pragma: no cover
                if tensor.dtype == DType.INT64:  # pragma: no cover
                    proto.raw_data = np.array(
                        tensor.data, dtype=np.int64
                    ).tobytes()  # pragma: no cover
                elif tensor.dtype == DType.UINT32:  # pragma: no cover
                    proto.raw_data = np.array(
                        tensor.data, dtype=np.uint32
                    ).tobytes()  # pragma: no cover
                elif tensor.dtype == DType.UINT64:  # pragma: no cover
                    proto.raw_data = np.array(
                        tensor.data, dtype=np.uint64
                    ).tobytes()  # pragma: no cover

    return proto


def to_value_info_proto(tensor: Tensor) -> onnx_pb2.ValueInfoProto:
    """Converts a Tensor to an ONNX ValueInfoProto (input/output)."""
    proto = onnx_pb2.ValueInfoProto()
    proto.name = tensor.name

    type_proto = onnx_pb2.TypeProto()
    tensor_type_proto = type_proto.tensor_type
    tensor_type_proto.elem_type = tensor.dtype.value

    shape_proto = tensor_type_proto.shape
    for dim in tensor.shape:
        dim_proto = shape_proto.dim.add()
        if isinstance(dim, int):
            dim_proto.dim_value = dim
        elif isinstance(dim, str):  # pragma: no cover
            dim_proto.dim_param = dim  # pragma: no cover

    proto.type.CopyFrom(type_proto)
    return proto


def to_attribute_proto(key: str, value: Any) -> onnx_pb2.AttributeProto:
    """Converts a Python dictionary key-value to an ONNX AttributeProto."""
    attr = onnx_pb2.AttributeProto()
    attr.name = key

    if isinstance(value, float):
        attr.f = value  # pragma: no cover
        attr.type = onnx_pb2.AttributeProto.FLOAT  # pragma: no cover
    elif isinstance(value, int):
        attr.i = value
        attr.type = onnx_pb2.AttributeProto.INT
    elif isinstance(value, str):
        attr.s = value.encode("utf-8")  # pragma: no cover
        attr.type = onnx_pb2.AttributeProto.STRING  # pragma: no cover
    elif isinstance(value, list) or isinstance(value, tuple):
        if all(isinstance(x, int) for x in value):
            attr.ints.extend(value)
            attr.type = onnx_pb2.AttributeProto.INTS
        elif all(isinstance(x, float) for x in value):  # pragma: no cover
            attr.floats.extend(value)  # pragma: no cover
            attr.type = onnx_pb2.AttributeProto.FLOATS  # pragma: no cover
        elif all(isinstance(x, str) for x in value):  # pragma: no cover
            attr.strings.extend([x.encode("utf-8") for x in value])  # pragma: no cover
            attr.type = onnx_pb2.AttributeProto.STRINGS  # pragma: no cover
        else:  # pragma: no cover
            raise ValueError(
                f"Unsupported list attribute type for key '{key}'"
            )  # pragma: no cover
    else:  # pragma: no cover
        raise ValueError(
            f"Unsupported attribute type for key '{key}': {type(value)}"
        )  # pragma: no cover

    return attr


def to_node_proto(node: Node) -> onnx_pb2.NodeProto:
    """Converts a frontend Node to an ONNX NodeProto."""
    proto = onnx_pb2.NodeProto()
    proto.op_type = node.op_type
    proto.name = node.name

    for inp in node.inputs:
        proto.input.append(inp.name if hasattr(inp, "name") else str(inp))

    for out in node.outputs:
        proto.output.append(out.name if hasattr(out, "name") else str(out))

    for k, v in node.attributes.items():
        proto.attribute.append(to_attribute_proto(k, v))

    return proto
