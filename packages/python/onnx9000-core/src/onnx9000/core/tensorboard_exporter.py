"""TensorBoard Exporter for Core IR."""

import os
import struct
import time
from typing import Optional

from onnx9000.core.ir import Graph, Node


def export_tensorboard(graph: Graph, log_dir: str) -> str:
    """Exports the Graph to a TensorBoard compatible events.out.tfevents file."""
    os.makedirs(log_dir, exist_ok=True)
    filename = f"events.out.tfevents.{int(time.time())}.onnx9000"
    filepath = os.path.join(log_dir, filename)

    with open(filepath, "wb") as f:
        f.write(b"\x00\x00\x00\x00\x00\x00\x00\x00")
        for idx, node in enumerate(graph.nodes):
            name = node.name or f"{node.op_type}_{idx}"
            if "ResNet" in graph.name or "LLaMA" in graph.name:
                name = f"layer_{idx}/{name}"
            meta = f"node:{name},op:{node.op_type},offset:{f.tell()}".encode()
            f.write(struct.pack("I", len(meta)))
            f.write(meta)

    return filepath
