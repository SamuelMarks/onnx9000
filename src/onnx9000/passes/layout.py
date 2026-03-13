"""
Memory Layout Transformations

Handles transpositions between NCHW (standard PyTorch/ONNX) and NHWC
(preferred by WebGL/WebGPU and specific hardware accelerators).
"""

from onnx9000.ir import Graph


def transform_nchw_to_nhwc(graph: Graph) -> None:
    """Converts standard NCHW convolutions/poolings to NHWC."""
    pass


def transform_nhwc_to_nchw(graph: Graph) -> None:
    """Converts NHWC back to NCHW."""
    pass
