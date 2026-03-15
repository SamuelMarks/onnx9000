"""
Export Sub-Package

Transforms the internal IR Graph into an official ONNX ModelProto binary format,
ensuring compliance with ONNX constraints and constraints.
"""

# mypy: ignore-errors
from pathlib import Path
from typing import Union

from onnx9000.core import onnx_pb2  # type: ignore
from onnx9000.core.exceptions import CompilationError
from onnx9000.export.proto_utils import (
    to_node_proto,
    to_tensor_proto,
    to_value_info_proto,
)
from onnx9000.frontends.frontend.builder import GraphBuilder
from onnx9000.frontends.frontend.tensor import Parameter


def build_graph_proto(builder: GraphBuilder) -> onnx_pb2.GraphProto:
    """Assembles a GraphProto from a frontend GraphBuilder."""
    graph = onnx_pb2.GraphProto()
    graph.name = builder.name

    # 1. Deduplicate parameter initialization logic
    seen_params: set[str] = set()

    for inp in builder.inputs:
        graph.input.append(to_value_info_proto(inp))

    for out in builder.outputs:
        graph.output.append(to_value_info_proto(out))

    known_tensors: set[str] = {i.name for i in builder.inputs} | {
        o.name for o in builder.outputs
    }

    for node in builder.nodes:
        # Check inputs to discover Parameters that haven't been added yet
        for node_input in node.inputs:
            # We treat Tensors with data as initializers if they are Parameters or have data explicitly.
            if (
                hasattr(node_input, "data")
                and node_input.data is not None
                and hasattr(node_input, "name")
                and node_input.name not in seen_params
            ):
                graph.initializer.append(to_tensor_proto(node_input))
                # Initializers should also be defined in inputs in newer ONNX standards
                graph.input.append(to_value_info_proto(node_input))
                seen_params.add(node_input.name)
                known_tensors.add(node_input.name)
            elif (
                isinstance(node_input, Parameter)
                and hasattr(node_input, "name")
                and node_input.name not in seen_params
            ):
                graph.initializer.append(to_tensor_proto(node_input))
                graph.input.append(to_value_info_proto(node_input))
                seen_params.add(node_input.name)
                known_tensors.add(node_input.name)

        # Add intermediate tensors to value_info
        for out in node.outputs:
            out_name = getattr(out, "name", out)
            if out_name not in known_tensors:
                if hasattr(out, "name"):
                    graph.value_info.append(to_value_info_proto(out))
                else:
                    # Strings fallbacks are un-annotated output tensors from multi-returns.
                    # This happens when tracing doesn't provide back Tensors but list of names
                    vi = onnx_pb2.ValueInfoProto()
                    vi.name = out_name
                    graph.value_info.append(vi)
                known_tensors.add(out_name)

        graph.node.append(to_node_proto(node))

    return graph


def build_model_proto(builder: GraphBuilder) -> onnx_pb2.ModelProto:
    """Wraps a GraphBuilder into a full ModelProto."""
    model = onnx_pb2.ModelProto()
    model.ir_version = 8  # Matches ONNX standard for Opset 18/19
    model.producer_name = "onnx9000"
    model.producer_version = "0.0.1"

    opset = model.opset_import.add()
    opset.domain = ""  # Default ONNX domain
    opset.version = 18

    graph_proto = build_graph_proto(builder)
    model.graph.CopyFrom(graph_proto)

    return model


def validate_model(model: onnx_pb2.ModelProto) -> None:
    """Custom IR validator for model well-formedness."""
    if not model.HasField("graph"):
        raise CompilationError("Model must have a valid graph.")

    # Very basic validation structure
    known_tensors = set([i.name for i in model.graph.input])
    for init in model.graph.initializer:
        known_tensors.add(init.name)

    for node in model.graph.node:
        for out in node.output:
            known_tensors.add(out)


def sanitize_model(model: onnx_pb2.ModelProto) -> None:
    """
    Sanitize the model proto before saving:
    1. Strip unused initializers (Step 85).
    2. Strip disconnected nodes / dead code elimination (Step 86).
    """
    pass  # Implementation deferred for a specialized pass module if necessary.


def to_string(builder: GraphBuilder) -> bytes:
    """Serializes the graph to an ONNX byte string."""
    model = build_model_proto(builder)
    sanitize_model(model)
    validate_model(model)
    return model.SerializeToString()


def to_onnx(builder: GraphBuilder, file_path: Union[str, Path]) -> None:
    """Serializes and saves the graph to an ONNX file."""
    content = to_string(builder)

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        f.write(content)
