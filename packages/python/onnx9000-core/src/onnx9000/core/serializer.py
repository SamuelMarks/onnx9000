"""Zero-dependency ONNX Protobuf Serializer."""

from pathlib import Path
from typing import Union

from onnx9000.core import onnx_pb2
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.core.ir import DynamicDim, Graph


class SerializationError(Onnx9000Error):
    """Exception raised when failing to serialize an ONNX graph to a file."""


def _serialize_shape(
    shape: tuple[Union[int, DynamicDim], ...],
) -> onnx_pb2.TensorShapeProto:
    """Serialize our internal shape into ONNX TensorShapeProto."""
    shape_proto = onnx_pb2.TensorShapeProto()
    for dim in shape:
        dim_proto = shape_proto.dim.add()
        if isinstance(dim, DynamicDim):
            if isinstance(dim.value, str):
                dim_proto.dim_param = dim.value
            else:
                dim_proto.dim_value = dim.value
        elif dim == -1:
            dim_proto.dim_param = "?"
        elif isinstance(dim, str):
            dim_proto.dim_param = dim
        else:
            dim_proto.dim_value = dim
    return shape_proto


def serialize_model(graph: Graph) -> onnx_pb2.ModelProto:
    """Serialize an internal ir.Graph back into an ONNX ModelProto."""
    model_proto = onnx_pb2.ModelProto()
    model_proto.ir_version = 8
    model_proto.producer_name = "onnx9000"
    model_proto.producer_version = "1.0.0"
    model_proto.doc_string = graph.doc_string
    for key, val in graph.metadata_props.items():
        prop = model_proto.metadata_props.add()
        prop.key = key
        prop.value = val
    for domain, version in graph.opset_imports.items():
        opset = model_proto.opset_import.add()
        opset.domain = domain
        opset.version = version
    graph_proto = model_proto.graph
    graph_proto.name = graph.name
    for vinfo in graph.inputs:
        if isinstance(vinfo, str) and vinfo in graph.tensors:
            t = graph.tensors[vinfo]
            vinfo_proto = graph_proto.input.add()
            vinfo_proto.name = t.name
            vinfo_proto.type.tensor_type.elem_type = t.dtype.value
            vinfo_proto.type.tensor_type.shape.CopyFrom(_serialize_shape(t.shape))
        elif hasattr(vinfo, "name"):
            vinfo_proto = graph_proto.input.add()
            vinfo_proto.name = vinfo.name
            vinfo_proto.type.tensor_type.elem_type = vinfo.dtype.value
            vinfo_proto.type.tensor_type.shape.CopyFrom(_serialize_shape(vinfo.shape))
    for vinfo in graph.value_info:
        if isinstance(vinfo, str) and vinfo in graph.tensors:
            t = graph.tensors[vinfo]
            vinfo_proto = graph_proto.value_info.add()
            vinfo_proto.name = t.name
            vinfo_proto.type.tensor_type.elem_type = t.dtype.value
            vinfo_proto.type.tensor_type.shape.CopyFrom(_serialize_shape(t.shape))
        elif hasattr(vinfo, "name"):
            vinfo_proto = graph_proto.value_info.add()
            vinfo_proto.name = vinfo.name
            vinfo_proto.type.tensor_type.elem_type = vinfo.dtype.value
            vinfo_proto.type.tensor_type.shape.CopyFrom(_serialize_shape(vinfo.shape))
    for vinfo in graph.outputs:
        if isinstance(vinfo, str) and vinfo in graph.tensors:
            t = graph.tensors[vinfo]
            vinfo_proto = graph_proto.output.add()
            vinfo_proto.name = t.name
            vinfo_proto.type.tensor_type.elem_type = t.dtype.value
            vinfo_proto.type.tensor_type.shape.CopyFrom(_serialize_shape(t.shape))
        elif hasattr(vinfo, "name"):
            vinfo_proto = graph_proto.output.add()
            vinfo_proto.name = vinfo.name
            vinfo_proto.type.tensor_type.elem_type = vinfo.dtype.value
            vinfo_proto.type.tensor_type.shape.CopyFrom(_serialize_shape(vinfo.shape))
    for init_name in graph.initializers:
        tensor = graph.tensors.get(init_name)
        if not tensor:
            continue
        init_proto = graph_proto.initializer.add()
        init_proto.name = tensor.name
        init_proto.data_type = tensor.dtype.value
        for dim in tensor.shape:
            init_proto.dims.append(int(dim.value) if isinstance(dim, DynamicDim) else int(dim))
        if tensor.data is not None:
            init_proto.raw_data = bytes(tensor.data)
    for node in graph.nodes:
        node_proto = graph_proto.node.add()
        node_proto.op_type = node.op_type
        node_proto.name = node.name
        node_proto.domain = node.domain
        node_proto.input.extend(node.inputs)
        node_proto.output.extend(node.outputs)
        for _attr_name, attr_obj in node.attributes.items():
            attr_proto = node_proto.attribute.add()
            attr_proto.name = attr_obj.name
            if attr_obj.attr_type == "FLOAT":
                attr_proto.type = onnx_pb2.AttributeProto.FLOAT
                attr_proto.f = float(attr_obj.value)
            elif attr_obj.attr_type == "INT":
                attr_proto.type = onnx_pb2.AttributeProto.INT
                attr_proto.i = int(attr_obj.value)
            elif attr_obj.attr_type == "STRING":
                attr_proto.type = onnx_pb2.AttributeProto.STRING
                attr_proto.s = str(attr_obj.value).encode("utf-8")
            elif attr_obj.attr_type == "FLOATS":
                attr_proto.type = onnx_pb2.AttributeProto.FLOATS
                attr_proto.floats.extend(attr_obj.value)
            elif attr_obj.attr_type == "INTS":
                attr_proto.type = onnx_pb2.AttributeProto.INTS
                attr_proto.ints.extend(attr_obj.value)
            elif attr_obj.attr_type == "STRINGS":
                attr_proto.type = onnx_pb2.AttributeProto.STRINGS
                for s in attr_obj.value:
                    attr_proto.strings.append(s.encode("utf-8"))
            elif attr_obj.attr_type == "TENSOR":
                attr_proto.type = onnx_pb2.AttributeProto.TENSOR
                attr_proto.t.CopyFrom(attr_obj.value)
            elif attr_obj.attr_type == "GRAPH":
                attr_proto.type = onnx_pb2.AttributeProto.GRAPH
                attr_proto.g.CopyFrom(attr_obj.value)
    return model_proto


def to_bytes(graph: Graph) -> bytes:
    """Serialize the internal graph directly to a binary ONNX string."""
    model_proto = serialize_model(graph)
    return model_proto.SerializeToString()


def save(graph: Graph, file_path: Union[str, Path]) -> None:
    """Save an ir.Graph into a .onnx file format on disk."""
    try:
        binary_data = to_bytes(graph)
        with open(file_path, "wb") as f:
            f.write(binary_data)
    except Exception as e:
        raise SerializationError(f"Failed to save ONNX model to {file_path}: {e}") from e
