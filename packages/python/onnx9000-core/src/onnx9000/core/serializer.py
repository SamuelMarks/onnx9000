"""Zero-dependency ONNX Protobuf Serializer."""

import logging
import zipfile
from pathlib import Path
from typing import Optional, Union

from onnx9000.core import onnx_pb2

logger = logging.getLogger(__name__)
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.core.ir import DynamicDim, Graph, SparseTensor, Tensor


class SerializationError(Onnx9000Error):
    """Exception raised for serialization errors."""


def _serialize_shape(shape: tuple[Union[int, str, DynamicDim], ...]) -> onnx_pb2.TensorShapeProto:
    """Serialize our internal shape into ONNX TensorShapeProto."""
    shape_proto = onnx_pb2.TensorShapeProto()
    for dim in shape:
        dim_proto = shape_proto.dim.add()
        if isinstance(dim, str):
            dim_proto.dim_param = dim
        elif isinstance(dim, DynamicDim):
            if isinstance(dim.value, int) and dim.value >= 0:
                dim_proto.dim_value = dim.value
            else:
                dim_proto.dim_param = str(dim.value)
        elif isinstance(dim, int):
            if dim < 0:
                dim_proto.dim_param = "?"
            else:
                dim_proto.dim_value = int(dim)
        else:
            if dim < 0:
                dim_proto.dim_param = "?"
            else:
                dim_proto.dim_value = int(dim)
    return shape_proto


def _serialize_tensor(
    tensor: "Tensor",
    base_dir: Optional[Path] = None,
    external_file: Optional[str] = None,
    offset: int = 0,
) -> tuple[onnx_pb2.TensorProto, int]:
    """Serialize our internal Tensor into ONNX TensorProto."""
    init_proto = onnx_pb2.TensorProto()
    init_proto.name = tensor.name
    init_proto.data_type = tensor.dtype.value
    for dim in tensor.shape:
        init_proto.dims.append(int(dim.value) if isinstance(dim, DynamicDim) else int(dim))

    if tensor.data is not None:
        raw_data = bytes(tensor.data)
        if external_file and len(raw_data) > 1024:  # Only externalize large-ish tensors
            init_proto.data_location = onnx_pb2.TensorProto.EXTERNAL
            entry = init_proto.external_data.add()
            entry.key = "location"
            entry.value = external_file
            entry = init_proto.external_data.add()
            entry.key = "offset"
            entry.value = str(offset)
            entry = init_proto.external_data.add()
            entry.key = "length"
            entry.value = str(len(raw_data))

            with open(base_dir / external_file, "ab") as f:
                f.write(raw_data)
            return init_proto, offset + len(raw_data)
        else:
            init_proto.raw_data = raw_data

    return init_proto, offset


def _serialize_sparse_tensor(
    sparse_tensor: "SparseTensor",
    base_dir: Optional[Path] = None,
    external_file: Optional[str] = None,
    offset: int = 0,
) -> tuple[onnx_pb2.SparseTensorProto, int]:
    """Serialize our internal SparseTensor into ONNX SparseTensorProto."""
    from onnx9000.core.sparse import sparse_to_coo

    coo_tensor = sparse_to_coo(sparse_tensor)

    sparse_proto = onnx_pb2.SparseTensorProto()
    current_offset = offset

    if coo_tensor.values:
        val_proto, current_offset = _serialize_tensor(
            coo_tensor.values, base_dir, external_file, current_offset
        )
        sparse_proto.values.CopyFrom(val_proto)
    if coo_tensor.indices:
        idx_proto, current_offset = _serialize_tensor(
            coo_tensor.indices, base_dir, external_file, current_offset
        )
        sparse_proto.indices.CopyFrom(idx_proto)

    for dim in coo_tensor.shape:
        sparse_proto.dims.append(int(dim.value) if isinstance(dim, DynamicDim) else int(dim))
    return sparse_proto, current_offset


def _sanitize_string(s: str) -> str:
    """Item 208: Sanitize ONNX strings natively during metadata packing."""
    return "".join(c for c in s if c.isprintable())


def serialize_model(
    graph: Graph, base_dir: Optional[Path] = None, external_file: Optional[str] = None
) -> onnx_pb2.ModelProto:
    """Serialize an internal ir.Graph back into an ONNX ModelProto."""
    model_proto = onnx_pb2.ModelProto()
    model_proto.ir_version = 8
    model_proto.producer_name = "onnx9000"
    model_proto.producer_version = "1.0.0"
    model_proto.doc_string = _sanitize_string(graph.doc_string)

    for key, val in graph.metadata_props.items():
        prop = model_proto.metadata_props.add()
        prop.key = _sanitize_string(key)
        prop.value = _sanitize_string(val)

    for domain, version in graph.opset_imports.items():
        opset = model_proto.opset_import.add()
        opset.domain = domain
        opset.version = version

    graph_proto = model_proto.graph
    graph_proto.name = graph.name

    def _serialize_vi(vi_obj: Union[str, "ValueInfo"]) -> onnx_pb2.ValueInfoProto:
        """Provides functional implementation."""

        vi_proto = onnx_pb2.ValueInfoProto()
        if isinstance(vi_obj, str):
            vi_proto.name = vi_obj
            tensor = graph.tensors.get(vi_obj)
            if tensor:
                vi_proto.type.tensor_type.elem_type = tensor.dtype.value
                vi_proto.type.tensor_type.shape.CopyFrom(_serialize_shape(tensor.shape))
        else:
            vi_proto.name = vi_obj.name
            vi_proto.type.tensor_type.elem_type = vi_obj.dtype.value
            vi_proto.type.tensor_type.shape.CopyFrom(_serialize_shape(vi_obj.shape))
        return vi_proto

    for vi in graph.inputs:
        graph_proto.input.append(_serialize_vi(vi))
    for vi in graph.outputs:
        graph_proto.output.append(_serialize_vi(vi))
    for vi in graph.value_info:
        graph_proto.value_info.append(_serialize_vi(vi))

    current_offset = 0
    if external_file and base_dir:
        # Clear external file if it exists
        if (base_dir / external_file).exists():
            (base_dir / external_file).unlink()

    for init_name in graph.initializers:
        tensor = graph.tensors.get(init_name)
        if not tensor:
            continue
        tensor_proto, current_offset = _serialize_tensor(
            tensor, base_dir, external_file, current_offset
        )
        graph_proto.initializer.append(tensor_proto)

    for sparse_init_name in graph.sparse_initializers:
        sparse_tensor = graph.tensors.get(sparse_init_name)
        if not isinstance(sparse_tensor, SparseTensor):
            continue
        sparse_proto, current_offset = _serialize_sparse_tensor(
            sparse_tensor, base_dir, external_file, current_offset
        )
        graph_proto.sparse_initializer.append(sparse_proto)

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
                attr_proto.f = float(attr_obj.value)
                attr_proto.type = onnx_pb2.AttributeProto.FLOAT
            elif attr_obj.attr_type == "INT":
                attr_proto.i = int(attr_obj.value)
                attr_proto.type = onnx_pb2.AttributeProto.INT
            elif attr_obj.attr_type == "STRING":
                attr_proto.s = str(attr_obj.value).encode("utf-8")
                attr_proto.type = onnx_pb2.AttributeProto.STRING
            elif attr_obj.attr_type == "FLOATS":
                attr_proto.floats.extend([float(v) for v in attr_obj.value])
                attr_proto.type = onnx_pb2.AttributeProto.FLOATS
            elif attr_obj.attr_type == "INTS":
                attr_proto.ints.extend([int(v) for v in attr_obj.value])
                attr_proto.type = onnx_pb2.AttributeProto.INTS
            elif attr_obj.attr_type == "STRINGS":
                attr_proto.strings.extend([str(v).encode("utf-8") for v in attr_obj.value])
                attr_proto.type = onnx_pb2.AttributeProto.STRINGS
            elif attr_obj.attr_type == "TENSOR":
                if isinstance(attr_obj.value, Tensor):
                    tensor_proto, _ = _serialize_tensor(attr_obj.value)
                    attr_proto.t.CopyFrom(tensor_proto)
                else:
                    attr_proto.t.CopyFrom(attr_obj.value)
                attr_proto.type = onnx_pb2.AttributeProto.TENSOR
            elif attr_obj.attr_type == "GRAPH":
                if isinstance(attr_obj.value, Graph):
                    attr_proto.g.CopyFrom(serialize_model(attr_obj.value).graph)
                else:
                    attr_proto.g.CopyFrom(attr_obj.value)
                attr_proto.type = onnx_pb2.AttributeProto.GRAPH

    return model_proto


def save(
    graph: Graph, file_path: Union[str, Path], external_data: bool = False, compress: bool = False
) -> None:
    """Save an internal ir.Graph to an ONNX file."""
    file_path = Path(file_path)
    base_dir = file_path.parent
    model_name = file_path.name

    external_file = None
    if external_data:
        external_file = model_name + ".data"

    try:
        model_proto = serialize_model(graph, base_dir, external_file)
        binary_data = model_proto.SerializeToString()

        if compress:
            # Item 138: Apply ZIP compression optionally
            zip_path = file_path.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(model_name, binary_data)
                if external_file and (base_dir / external_file).exists():
                    zf.write(base_dir / external_file, external_file)
            logger.info(f"Compressed model saved to {zip_path}")
        else:
            with open(file_path, "wb") as f:
                f.write(binary_data)
            logger.info(f"Model saved to {file_path}")

    except Exception as e:
        raise SerializationError(f"Failed to save ONNX model to {file_path}: {e}") from e


def to_bytes(graph: Graph) -> bytes:
    """Serialize an internal ir.Graph to ONNX binary format."""
    return serialize_model(graph).SerializeToString()
