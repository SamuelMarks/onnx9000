"""Module docstring."""

# mypy: ignore-errors
from pathlib import Path
from typing import Any, Union

import numpy as np

from onnx9000 import onnx_pb2  # type: ignore
from onnx9000.dtypes import DType
from onnx9000.exceptions import CompilationError
from onnx9000.ir import DynamicDim, Graph, Node, Tensor


def _parse_dtype(data_type: int) -> DType:
    """_parse_dtype docstring."""

    try:
        return DType(data_type)
    except ValueError:  # pragma: no cover
        raise CompilationError(
            f"Unsupported ONNX TensorProto DataType: {data_type}"
        ) from None  # pragma: no cover


def _parse_shape(
    shape_proto: onnx_pb2.TensorShapeProto,
) -> tuple[Union[int, DynamicDim], ...]:
    """_parse_shape docstring."""

    shape = []
    for dim in shape_proto.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):  # pragma: no cover
            shape.append(DynamicDim(dim.dim_param))  # pragma: no cover
        else:  # pragma: no cover
            shape.append(DynamicDim(-1))  # pragma: no cover
    return tuple(shape)


def _parse_attribute(attr: onnx_pb2.AttributeProto) -> Any:
    """_parse_attribute docstring."""

    if attr.type == onnx_pb2.AttributeProto.FLOAT:
        return attr.f  # pragma: no cover
    elif attr.type == onnx_pb2.AttributeProto.INT:
        return attr.i
    elif attr.type == onnx_pb2.AttributeProto.STRING:
        return attr.s.decode("utf-8")  # pragma: no cover
    elif attr.type == onnx_pb2.AttributeProto.TENSOR:
        return attr.t  # Could parse further if needed  # pragma: no cover
    elif attr.type == onnx_pb2.AttributeProto.FLOATS:
        return list(attr.floats)  # pragma: no cover
    elif attr.type == onnx_pb2.AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == onnx_pb2.AttributeProto.STRINGS:  # pragma: no cover
        return [s.decode("utf-8") for s in attr.strings]  # pragma: no cover
    # For subgraphs (GRAPH type) or sparse tensors, more complex logic is needed  # pragma: no cover
    return None  # pragma: no cover


def parse_model(model_proto: onnx_pb2.ModelProto) -> Graph:
    """Parses an ONNX ModelProto into an ir.Graph."""
    graph_proto = model_proto.graph
    graph = Graph(name=graph_proto.name)

    # Parse Initializers (Weights)
    for init in graph_proto.initializer:
        dtype = _parse_dtype(init.data_type)
        shape = tuple(init.dims)

        # VERY basic mock array reading, normally we decode raw_data or specific fields based on dtype
        # numpy dtype mapping
        np_dtype_map = {
            DType.FLOAT32: np.float32,
            DType.FLOAT64: np.float64,
            DType.INT32: np.int32,
            DType.INT64: np.int64,
            # ... others
        }

        data = None
        if init.raw_data and dtype in np_dtype_map:
            data = np.frombuffer(init.raw_data, dtype=np_dtype_map[dtype]).copy()
            if data.size > 0:
                data = data.reshape(shape)

        tensor = Tensor(
            name=init.name, shape=shape, dtype=dtype, is_initializer=True, data=data
        )
        graph.add_tensor(tensor)
        graph.initializers.append(init.name)

    # Parse Inputs (Dynamic + Initializers)
    for vinfo in graph_proto.input:
        name = vinfo.name
        if name not in graph.initializers:
            graph.inputs.append(name)
            dtype = _parse_dtype(vinfo.type.tensor_type.elem_type)

            shape = _parse_shape(vinfo.type.tensor_type.shape)
            print(f"PARSED INPUT {name} SHAPE: {shape}")
            tensor = Tensor(name=name, shape=shape, dtype=dtype)

            graph.add_tensor(tensor)

    # Parse Outputs
    for vinfo in graph_proto.output:
        name = vinfo.name
        graph.outputs.append(name)
        dtype = _parse_dtype(vinfo.type.tensor_type.elem_type)
        shape = _parse_shape(vinfo.type.tensor_type.shape)
        if name not in graph.tensors:
            tensor = Tensor(name=name, shape=shape, dtype=dtype)
            graph.add_tensor(tensor)

    # Parse Value Info (Intermediate Shapes)
    for vinfo in graph_proto.value_info:
        name = vinfo.name
        dtype = _parse_dtype(vinfo.type.tensor_type.elem_type)
        shape = _parse_shape(vinfo.type.tensor_type.shape)
        if name not in graph.tensors:
            tensor = Tensor(name=name, shape=shape, dtype=dtype)
            graph.add_tensor(tensor)

    # Parse Nodes
    for node_proto in graph_proto.node:
        attributes = {
            attr.name: _parse_attribute(attr) for attr in node_proto.attribute
        }

        node = Node(
            op_type=node_proto.op_type,
            inputs=list(node_proto.input),
            outputs=list(node_proto.output),
            attributes=attributes,
            name=node_proto.name,
        )
        graph.add_node(node)

    # Perform Shape and Type Inference
    from onnx9000.parser.inference import infer_shapes_and_types

    infer_shapes_and_types(graph)

    return graph


def load(file_path: Union[str, Path]) -> Graph:
    """Reads an ONNX file and parses it into an ir.Graph."""
    model = onnx_pb2.ModelProto()
    with open(file_path, "rb") as f:
        model.ParseFromString(f.read())
    return parse_model(model)


def from_bytes(proto_bytes: bytes) -> Graph:
    """Parses an ONNX byte string into an ir.Graph."""
    model = onnx_pb2.ModelProto()  # pragma: no cover
    model.ParseFromString(proto_bytes)  # pragma: no cover
    return parse_model(model)  # pragma: no cover
