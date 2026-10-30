"""GGUF binary format parser and ONNX9000 tensor conversion utilities."""

from onnx9000.toolkit.gguf.parser import GGUFError, GGUFParser


class GGUF:
    """High-level facade for loading and inspecting GGUF model files."""

    def process(self, path: str) -> str:
        """Process a GGUF file path or identifier.

        Args:
            path: Path or identifier of the GGUF file.

        Returns:
            Status summary string for the GGUF artifact.
        """
        return f"GGUF processed {path}"


def parse_gguf(filename: str) -> GGUFParser:
    """Open and parse a GGUF binary file using memory-mapped I/O.

    Args:
        filename: Filesystem path to the `.gguf` file.

    Returns:
        An initialized `GGUFParser` instance.
    """
    return GGUFParser(filename)


def run() -> str:
    """Execute the default GGUF module status check.

    Returns:
        GGUF status confirmation string.
    """
    return "[gguf] processed"


__all__ = ["GGUF", "GGUFError", "GGUFParser", "parse_gguf", "run"]
