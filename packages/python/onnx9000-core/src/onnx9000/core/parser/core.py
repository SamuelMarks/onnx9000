"""Module providing core logic and structural definitions for ONNX parsing."""

import struct
from pathlib import Path
from typing import Any, Optional, Union

from onnx9000.core import onnx_pb2
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import ONNXParseError
from onnx9000.core.ir import (
    Attribute,
    Constant,
    DynamicDim,
    Graph,
    Node,
    Tensor,
    ValueInfo,
    Variable,
)


def _parse_dtype(data_type: int) -> DType:
    """Parse ONNX data type."""
    try:
        return DType(data_type)
    except ValueError:
        raise ONNXParseError(f"Unsupported ONNX TensorProto DataType: {data_type}") from None


def _parse_shape(shape_proto: Any) -> tuple[Union[int, DynamicDim], ...]:
    """Parse ONNX shape proto."""
    shape: list[Union[int, DynamicDim]] = []
    for dim in shape_proto.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(DynamicDim(dim.dim_param))
        else:
            shape.append(DynamicDim(-1))
    return tuple(shape)


def _parse_attribute(attr: Any) -> Attribute:
    """Parse ONNX attribute proto."""
    if attr.type == onnx_pb2.AttributeProto.FLOAT:
        return Attribute(attr.name, "FLOAT", attr.f)
    elif attr.type == onnx_pb2.AttributeProto.INT:
        return Attribute(attr.name, "INT", attr.i)
    elif attr.type == onnx_pb2.AttributeProto.STRING:
        return Attribute(attr.name, "STRING", attr.s.decode("utf-8"))
    elif attr.type == onnx_pb2.AttributeProto.TENSOR:
        return Attribute(attr.name, "TENSOR", attr.t)
    elif attr.type == onnx_pb2.AttributeProto.GRAPH:
        return Attribute(attr.name, "GRAPH", attr.g)
    elif attr.type == onnx_pb2.AttributeProto.FLOATS:
        return Attribute(attr.name, "FLOATS", list(attr.floats))
    elif attr.type == onnx_pb2.AttributeProto.INTS:
        return Attribute(attr.name, "INTS", list(attr.ints))
    elif attr.type == onnx_pb2.AttributeProto.STRINGS:
        return Attribute(attr.name, "STRINGS", [s.decode("utf-8") for s in attr.strings])
    return Attribute(attr.name, "UNKNOWN", None)


def parse_tensor_proto(init: Any, base_dir: Optional[Path] = None) -> Tensor:
    """Parse a single ONNX TensorProto into an ir.Tensor."""
    dtype = _parse_dtype(init.data_type)
    shape = tuple(init.dims)
    data: Optional[Union[bytes, memoryview]] = None

    if hasattr(init, "data_location") and init.data_location == 1:
        location = ""
        offset = 0
        length = 0
        for entry in init.external_data:
            if entry.key == "location":
                location = entry.value
            elif entry.key == "offset":
                offset = int(entry.value)
            elif entry.key == "length":
                length = int(entry.value)

        if location and base_dir:
            import mmap

            file_path = base_dir / location
            with open(file_path, "rb") as ef:
                mm = mmap.mmap(ef.fileno(), 0, access=mmap.ACCESS_READ)
                if length > 0:
                    data = memoryview(mm)[offset : offset + length]
                else:
                    data = memoryview(mm)[offset:]

    if data is None:
        if init.raw_data:
            data = memoryview(init.raw_data)
        elif len(init.float_data) > 0:
            data = memoryview(struct.pack(f"<{len(init.float_data)}f", *init.float_data))
        elif len(init.int32_data) > 0:
            data = memoryview(struct.pack(f"<{len(init.int32_data)}i", *init.int32_data))
        elif len(init.int64_data) > 0:
            data = memoryview(struct.pack(f"<{len(init.int64_data)}q", *init.int64_data))
        elif len(init.string_data) > 0:
            data = b"\x00".join(init.string_data)
    tensor = Constant(name=init.name, shape=shape, dtype=dtype, values=data)
    return tensor


def load_tensor(file_path: Union[str, Path]) -> Tensor:
    """Read an ONNX Tensor file (.pb) and parses it into an ir.Tensor."""
    tensor_proto = onnx_pb2.TensorProto()
    with open(file_path, "rb") as f:
        tensor_proto.ParseFromString(f.read())
    return parse_tensor_proto(tensor_proto)


def parse_model(model_proto: Any, base_dir: Optional[Path] = None) -> Graph:
    """Parse an ONNX ModelProto into an ir.Graph."""
    graph_proto = model_proto.graph
    graph = Graph(name=graph_proto.name)
    graph.doc_string = model_proto.doc_string
    graph.producer_name = model_proto.producer_name
    graph.producer_version = model_proto.producer_version
    for prop in model_proto.metadata_props:
        graph.metadata_props[prop.key] = prop.value
    for opset in model_proto.opset_import:
        graph.opset_imports[opset.domain] = opset.version
    for init in graph_proto.initializer:
        tensor = parse_tensor_proto(init, base_dir)
        graph.add_tensor(tensor)
        graph.initializers.append(init.name)
    for vinfo in graph_proto.input:
        name = vinfo.name
        if name not in graph.initializers:
            dtype = _parse_dtype(vinfo.type.tensor_type.elem_type)
            shape = _parse_shape(vinfo.type.tensor_type.shape)
            graph.inputs.append(ValueInfo(name, shape, dtype))
            tensor = Variable(name=name, shape=shape, dtype=dtype)
            graph.add_tensor(tensor)
    for vinfo in graph_proto.output:
        name = vinfo.name
        dtype = _parse_dtype(vinfo.type.tensor_type.elem_type)
        shape = _parse_shape(vinfo.type.tensor_type.shape)
        graph.outputs.append(ValueInfo(name, shape, dtype))
        if name not in graph.tensors:
            tensor = Variable(name=name, shape=shape, dtype=dtype)
            graph.add_tensor(tensor)
    for vinfo in graph_proto.value_info:
        name = vinfo.name
        dtype = _parse_dtype(vinfo.type.tensor_type.elem_type)
        shape = _parse_shape(vinfo.type.tensor_type.shape)
        graph.value_info.append(ValueInfo(name, shape, dtype))
        if name not in graph.tensors:
            tensor = Variable(name=name, shape=shape, dtype=dtype)
            graph.add_tensor(tensor)
    for node_proto in graph_proto.node:
        attributes = {attr.name: _parse_attribute(attr) for attr in node_proto.attribute}
        node = Node(
            op_type=node_proto.op_type,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            attributes=attributes,
            name=node_proto.name,
            domain=node_proto.domain,
        )
        graph.add_node(node)
    return graph


def load(file_path: Union[str, Path]) -> Graph:
    """Read an ONNX file and parses it into an ir.Graph."""
    path_obj = Path(file_path)
    with open(file_path, "rb") as f:
        data = f.read()

    # ONNX magic bytes validation for Protobuf
    if len(data) < 8:
        raise ONNXParseError("File is too small to be a valid ONNX protobuf")

    model = onnx_pb2.ModelProto()
    try:
        model.ParseFromString(data)
    except Exception as e:
        raise ONNXParseError(f"Failed to parse ONNX file: {e}") from e

    # Extra validation
    if model.ir_version == 0:
        raise ONNXParseError("Invalid ONNX file: ir_version is 0 or missing")

    return parse_model(model, path_obj.parent)


def from_bytes(proto_bytes: bytes) -> Graph:
    """Parse an ONNX byte string into an ir.Graph."""
    if len(proto_bytes) < 8:
        raise ONNXParseError("Byte string is too small to be a valid ONNX protobuf")

    model = onnx_pb2.ModelProto()
    try:
        model.ParseFromString(proto_bytes)
    except Exception as e:
        raise ONNXParseError(f"Failed to parse ONNX bytes: {e}") from e

    if model.ir_version == 0:
        raise ONNXParseError("Invalid ONNX bytes: ir_version is 0 or missing")

    return parse_model(model)
