"""Module providing core logic and structural definitions."""

import argparse
import sys
from onnx9000.extensions.text.exporter import export_tokenizer_binary


def main() -> None:
    """Provides semantic functionality and verification."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace tokenizer.json to ONNX9000 compressed format."
    )
    parser.add_argument("input", help="Path to input tokenizer.json")
    parser.add_argument("output", help="Path to output .bin file")

    args = parser.parse_args()

    try:
        export_tokenizer_binary(args.input, args.output)
        print(f"Successfully exported to {args.output}")
    except Exception as e:
        print(f"Failed to export: {e}", file=sys.stderr)
        sys.exit(1)
