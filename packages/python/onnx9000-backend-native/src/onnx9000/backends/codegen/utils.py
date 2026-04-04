"""C++ Code Generation Utilities.

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

import re


def get_attribute(node, name, default=None):
    """Get an attribute value from an ONNX node.

    Args:
        node: The ONNX Node.
        name: The name of the attribute.
        default: The default value if the attribute is not found.

    Returns:
        The attribute value or the default value.

    """
    if node.attributes and name in node.attributes:
        return node.attributes[name].value
    return default


def sanitize_name(name: str) -> str:
    """Sanitizes ONNX string IDs to make them valid C++ variable names.

    Replaces non-alphanumeric characters with underscores.
    Prepends `var_` if the name starts with a number.
    """
    clean = re.sub("[^a-zA-Z0-9_]", "_", name)
    if clean and clean[0].isdigit():
        clean = f"var_{clean}"
    return clean


def get_omp_pragma(total_elements: str, threshold: int = 10000, extra: str = "") -> str:
    """Return an OpenMP/SIMD pragma conditionally based on the target.

    In a full system, this would inspect if the target is WASM or Native.
    For Emscripten (WASM), we emit `#pragma clang loop vectorize(enable)`.
    For native, we emit `#pragma omp parallel for`.
    """
    from onnx9000.core import config

    if config.ONNX9000_USE_CUDA:
        return ""

    pragmas = []
    if getattr(config, "ONNX9000_ENABLE_LOOP_UNROLLING", False):
        pragmas.append("#pragma unroll")

    pragmas.append(
        f"#if defined(__EMSCRIPTEN__)\n#pragma clang loop vectorize(enable)\n#else\n#pragma omp parallel for {extra}\n#endif"
    )

    return "\n".join(pragmas)
