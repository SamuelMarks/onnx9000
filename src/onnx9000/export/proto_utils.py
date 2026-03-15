"""
Export Sub-Package

Transforms the internal IR Graph into an official ONNX ModelProto binary format,
ensuring compliance with ONNX constraints and constraints.
"""

# mypy: ignore-errors
from typing import Any

from onnx9000.core import onnx_pb2  # type: ignore
from onnx9000.frontends.frontend.tensor import Node, Parameter, Tensor


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
        elif isinstance(dim, str):
            dim_proto.dim_param = dim

    proto.type.CopyFrom(type_proto)
    return proto


def to_attribute_proto(key: str, value: Any) -> onnx_pb2.AttributeProto:
    """Converts a Python dictionary key-value to an ONNX AttributeProto."""
    attr = onnx_pb2.AttributeProto()
    attr.name = key

    if isinstance(value, float):
        attr.f = value
        attr.type = onnx_pb2.AttributeProto.FLOAT
    elif isinstance(value, int):
        attr.i = value
        attr.type = onnx_pb2.AttributeProto.INT
    elif isinstance(value, str):
        attr.s = value.encode("utf-8")
        attr.type = onnx_pb2.AttributeProto.STRING
    elif isinstance(value, list) or isinstance(value, tuple):
        if all(isinstance(x, int) for x in value):
            attr.ints.extend(value)
            attr.type = onnx_pb2.AttributeProto.INTS
        elif all(isinstance(x, float) for x in value):
            attr.floats.extend(value)
            attr.type = onnx_pb2.AttributeProto.FLOATS
        elif all(isinstance(x, str) for x in value):
            attr.strings.extend([x.encode("utf-8") for x in value])
            attr.type = onnx_pb2.AttributeProto.STRINGS
        else:
            raise ValueError(f"Unsupported list attribute type for key '{key}'")
    else:
        raise ValueError(f"Unsupported attribute type for key '{key}': {type(value)}")

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
