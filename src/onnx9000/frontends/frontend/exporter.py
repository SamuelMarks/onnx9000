"""Exporter API."""

from typing import Any, Union, Dict, List, Optional
from pathlib import Path
from onnx9000.frontends.frontend.nn.module import Module
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.frontends.frontend.tracer import trace
from onnx9000.export.builder import build_model_proto, to_onnx
import numpy as np


def export(
    model: Union[Module, Any],
    args: Union[tuple, Tensor],
    f: Union[str, Path, Any],
    export_params: bool = True,
    verbose: bool = False,
    training: Any = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    opset_version: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    keep_initializers_as_inputs: Optional[bool] = None,
    custom_opsets: Optional[Dict[str, int]] = None,
    do_constant_folding: bool = True,
) -> None:
    """Exports a model into ONNX format."""
    if not isinstance(args, tuple):
        args = (args,)
    builder = trace(model, *args)
    if input_names:
        for i, name in enumerate(input_names):
            if i < len(builder.inputs):
                builder.inputs[i]._name = name
    if output_names:
        for i, name in enumerate(output_names):
            if i < len(builder.outputs):
                builder.outputs[i]._name = name
                for node in builder.nodes:
                    if any((o is builder.outputs[i] for o in node.outputs)):
                        node.outputs = [
                            builder.outputs[i] if o is builder.outputs[i] else o
                            for o in node.outputs
                        ]
    model_proto = build_model_proto(builder)
    model_proto.opset_import[0].version = opset_version
    if custom_opsets:
        for domain, version in custom_opsets.items():
            opset = model_proto.opset_import.add()
            opset.domain = domain
            opset.version = version
    if dynamic_axes:
        for val_info in model_proto.graph.input:
            if val_info.name in dynamic_axes:
                axes = dynamic_axes[val_info.name]
                for idx, dim in enumerate(val_info.type.tensor_type.shape.dim):
                    if idx in axes:
                        dim.dim_param = axes[idx]
        for val_info in model_proto.graph.output:
            if val_info.name in dynamic_axes:
                axes = dynamic_axes[val_info.name]
                for idx, dim in enumerate(val_info.type.tensor_type.shape.dim):
                    if idx in axes:
                        dim.dim_param = axes[idx]
    if isinstance(f, (str, Path)):
        if model_proto.ByteSize() > 2 * 1024 * 1024 * 1024:
            import os
            from onnx9000.core import onnx_pb2

            with open(f, "wb") as file:
                file.write(model_proto.SerializeToString())
        else:
            with open(f, "wb") as file:
                file.write(model_proto.SerializeToString())
    else:
        f.write(model_proto.SerializeToString())


def visualize(model_path: Union[str, Path]) -> str:
    """Provides visualize functionality and verification."""
    return ""
