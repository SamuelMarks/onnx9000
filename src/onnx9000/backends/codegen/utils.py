"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

import re


def sanitize_name(name: str) -> str:
    """
    Sanitizes ONNX string IDs to make them valid C++ variable names.
    Replaces non-alphanumeric characters with underscores.
    Prepends 'var_' if the name starts with a number.
    """
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if clean and clean[0].isdigit():
        clean = f"var_{clean}"
    return clean


def get_omp_pragma(total_elements: str, threshold: int = 10000) -> str:
    """
    Returns an OpenMP/SIMD pragma conditionally based on the target.
    In a full system, this would inspect if the target is WASM or Native.
    For Emscripten (WASM), we emit `#pragma clang loop vectorize(enable)`.
    For native, we emit `#pragma omp parallel for`.
    """
    from onnx9000.core import config

    if config.ONNX9000_USE_CUDA:
        return ""  # CUDA uses explicit kernel launches, not OpenMP pragmas here

    return "#if defined(__EMSCRIPTEN__)\n        #pragma clang loop vectorize(enable)\n        #else\n        #pragma omp parallel for\n        #endif"
