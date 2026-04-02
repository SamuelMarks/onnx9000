"""Packaging and CLI bundling for ONNX C Code Generator."""

import os


def bundle_weights_bin(weights_data: bytes, output_dir: str, prefix: str):
    """Bundle external weight binaries into .bin files loaded via fopen."""
    bin_path = os.path.join(output_dir, f"{prefix}weights.bin")
    with open(bin_path, "wb") as f:
        f.write(weights_data)
    return bin_path


def generate_memory_summary(arena_size: int, num_nodes: int, num_tensors: int) -> str:
    """Print memory usage summary table as a C block comment at the top of the header file."""
    return f"""/*
 * ONNX9000 C99 Compiled Model Summary
 * -------------------------------------
 * Total Nodes:     {num_nodes}
 * Total Tensors:   {num_tensors}
 * Peak Arena RAM:  {arena_size} bytes
 */
"""
