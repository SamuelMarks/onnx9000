"""Command line interface for the ONNX9000 C89 compiler backend.

This tool converts ONNX models into standalone, dependency-free C89 code
suitable for embedded devices and restricted environments.
"""

import argparse
import os
import sys

from onnx9000.c_compiler.compiler import C89Compiler
from onnx9000.core.parser.core import load


def main():
    """Execute the C compiler CLI, parsing arguments and generating C code."""
    parser = argparse.ArgumentParser(
        description="onnx9000-c-compiler: Convert ONNX models to standalone C89 code."
    )
    parser.add_argument("input_model", help="Path to the input .onnx model.")
    parser.add_argument(
        "--output-dir", "-o", default="out_c", help="Output directory for generated C files."
    )
    parser.add_argument(
        "--prefix", default="model_", help="Prefix for generated C functions and structs."
    )
    parser.add_argument(
        "--target",
        default="",
        help="Target hardware architecture (e.g. 'cmsis-nn', 'esp-nn', 'arduino', 'baremetal', 'avx2').",
    )
    parser.add_argument(
        "--emit-cpp", action="store_true", help="Emit C++ compatible headers (extern C)."
    )
    parser.add_argument(
        "--no-math-h", action="store_true", help="Do not include <math.h> (uses inline fallbacks)."
    )

    parser.add_argument("--quiet", "-q", action="store_true", help="Limit output verbosity")
    parser.add_argument("--no-opt", action="store_true", help="Disable graph optimization")
    parser.add_argument("--align", type=int, default=0, help="Alignment (e.g. 16, 32)")
    parser.add_argument("--indent-spaces", type=int, default=4, help="Indentation spaces")
    args = parser.parse_args()

    if not os.path.exists(args.input_model):
        sys.exit(1)

    if not args.quiet:
        print(f"Loading ONNX model: {args.input_model}")
    with open(args.input_model, "rb") as f:
        f.read()

    # We should have a way to get the ONNX Graph, ideally parse_onnx handles this and returns an ONNX Graph object
    # For testing and compilation, we must convert it to IR Graph
    graph = load(args.input_model)

    if not args.quiet:
        print(f"Model loaded. Nodes: {len(graph.nodes)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Phase 18: Optimization Passes
    if not args.no_opt:
        if not args.quiet:
            print(
                "Warning: internal optimization passes in compiler are removed. Please run onnx9000-optimizer separately."
            )

    # Phase 18: Strip debug
    for t in graph.tensors.values():
        t.doc_string = ""
    graph.doc_string = ""

    compiler = C89Compiler(
        graph,
        prefix=args.prefix,
        emit_cpp=args.emit_cpp,
        target=args.target,
        use_math_h=not args.no_math_h
        and any(getattr(t, "dtype", None) in (1, 10, 16, 11) for t in graph.tensors.values()),
        align=args.align,
        indent=args.indent_spaces,
    )

    h_code, c_code = compiler.generate()

    h_path = os.path.join(args.output_dir, f"{args.prefix}model.h")
    c_path = os.path.join(args.output_dir, f"{args.prefix}model.c")

    with open(h_path, "w") as f:
        f.write(h_code)
    with open(c_path, "w") as f:
        f.write(c_code)

    if not args.quiet:
        print(f"Success! Generated C model in {args.output_dir}/")
    if not args.quiet:
        print(f"  Header: {h_path}")
    if not args.quiet:
        print(f"  Source: {c_path}")

    from onnx9000.c_compiler.bundler import generate_memory_summary

    # Prepend summary to header
    with open(h_path) as f:
        h_content = f.read()

    summary = generate_memory_summary(compiler.arena_size, len(graph.nodes), len(graph.tensors))

    with open(h_path, "w") as f:
        f.write(summary + "\n" + h_content)

    if args.target == "arduino":
        ino_path = os.path.join(args.output_dir, f"{args.prefix}sketch.ino")
        from onnx9000.c_compiler.project_generator import generate_arduino_sketch

        with open(ino_path, "w") as f:
            f.write(generate_arduino_sketch(args.prefix))
        if not args.quiet:
            print(f"  Arduino Sketch: {ino_path}")
    else:
        cm_path = os.path.join(args.output_dir, "CMakeLists.txt")
        from onnx9000.c_compiler.project_generator import generate_cmakelists

        with open(cm_path, "w") as f:
            f.write(generate_cmakelists(args.prefix))
        if not args.quiet:
            print(f"  CMakeLists: {cm_path}")

    # 285: JSON output metadata
    import json

    meta = {
        "prefix": args.prefix,
        "nodes": len(graph.nodes),
        "arena_size": compiler.arena_size,
        "inputs": [str(x) for x in graph.inputs],
        "outputs": [str(x) for x in graph.outputs],
    }
    with open(os.path.join(args.output_dir, f"{args.prefix}metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
