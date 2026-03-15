"""Module providing core logic and structural definitions."""

import argparse
import sys
from pathlib import Path

from onnx9000 import compile as core_compile
from onnx9000.utils.logger import logger


def main() -> None:
    """CLI entrypoint for onnx9000."""
    parser = argparse.ArgumentParser(
        prog="onnx9000",
        description="The JIT Transpiler Engine for ONNX. Compiles ONNX graphs to bespoke C++ or WebAssembly.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: compile
    compile_parser = subparsers.add_parser(
        "compile", help="Ahead-of-Time (AOT) compile an ONNX model to C++ or WASM."
    )
    compile_parser.add_argument(
        "model", type=Path, help="Path to the .onnx model file."
    )
    compile_parser.add_argument(
        "--target",
        type=str,
        choices=["cpp", "wasm"],
        default="cpp",
        help="Compilation target: 'cpp' (native Pybind11 extension) or 'wasm' (Emscripten).",
    )

    args = parser.parse_args()

    if args.command == "compile":
        logger.info(f"Compiling {args.model} for target {args.target}...")
        try:
            core_compile(str(args.model), target=args.target)
            logger.info("Compilation successful.")
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            sys.exit(1)
