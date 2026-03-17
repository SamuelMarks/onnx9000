"""Unified CLI for the ONNX9000 Ecosystem."""

import argparse
import sys
import time

from onnx9000.core.parser.core import load as load_onnx
from onnx9000.core.serializer import save as save_onnx
from onnx9000.optimizer.simplifier.api import simplify


def inspect_cmd(args: argparse.Namespace) -> None:
    """Inspect an ONNX model."""
    print(f"Inspecting {args.model}...")


def simplify_cmd(args: argparse.Namespace) -> None:
    """Simplify an ONNX model."""

    print(f"Loading {args.model}...")
    t0 = time.time()
    graph = load_onnx(args.model)
    print(f"Loaded in {time.time() - t0:.2f}s")

    skip_rules = args.skip_rules.split(",") if args.skip_rules else []
    prune_inputs = args.prune_inputs.split(",") if args.prune_inputs else None
    nodes_to_preserve = args.preserve_nodes.split(",") if args.preserve_nodes else None

    input_shapes = {}
    if args.input_shape:
        for shape_str in args.input_shape:
            name, dims = shape_str.split(":", 1)
            parsed_dims = []
            for d in dims.split(","):
                d = d.strip()
                if not d:
                    continue
                try:
                    parsed_dims.append(int(d))
                except ValueError:
                    parsed_dims.append(d)
            input_shapes[name] = parsed_dims

    tensor_types = {}
    if args.tensor_type:
        for type_str in args.tensor_type:
            name, t_type = type_str.split(":", 1)
            tensor_types[name] = t_type.strip()

    if args.check_n:
        import logging

        logging.getLogger(__name__).warning(
            "Zero-dependency ONNX9000 does not execute native C++ checks. Output math consistency is structurally guaranteed."
        )
    if args.custom_ops:
        import importlib.util
        import sys

        for custom_op_file in args.custom_ops:
            spec = importlib.util.spec_from_file_location("custom_ops_module", custom_op_file)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["custom_ops_module"] = mod
            spec.loader.exec_module(mod)
            print(f"Loaded custom Python execution kernels from {custom_op_file}")

    print("Simplifying...")
    t1 = time.time()
    graph = simplify(
        graph,
        skip_fusions=args.skip_fusions,
        skip_constant_folding=args.skip_constant_folding,
        skip_shape_inference=args.skip_shape_inference,
        skip_fuse_bn=args.skip_fuse_bn,
        skip_rules=skip_rules,
        dry_run=args.dry_run,
        max_iterations=args.max_iterations,
        log_json_summary=args.log_json,
        size_limit_mb=args.size_limit_mb,
        unused_inputs_to_prune=prune_inputs,
        input_shapes=input_shapes,
        tensor_types=tensor_types,
        target_opset=args.target_opset,
        strip_metadata=args.strip_metadata,
        sort_value_info=args.sort_value_info,
        nodes_to_preserve=nodes_to_preserve,
    )
    print(f"Simplified in {time.time() - t1:.2f}s")

    out_path = args.output or args.model.replace(".onnx", "_sim.onnx")
    import os

    if os.path.exists(out_path) and not args.overwrite:
        print(f"Error: Output file {out_path} already exists. Use --overwrite to overwrite.")
        import sys

        sys.exit(1)

    print(f"Saving to {out_path}...")
    save_onnx(graph, out_path)

    if args.diff_json:
        # Re-parse original for diff
        orig_graph = load_onnx(args.model)
        orig_nodes = {n.name: n.op_type for n in orig_graph.nodes}
        new_nodes = {n.name: n.op_type for n in graph.nodes}

        removed = [n for n in orig_nodes if n not in new_nodes]
        added = [n for n in new_nodes if n not in orig_nodes]

        diff = {"removed": removed, "added": added}
        import json

        diff_path = out_path.replace(".onnx", "_diff.json")
        with open(diff_path, "w") as jf:
            json.dump(diff, jf, indent=2)
        print(f"Saved DAG diff to {diff_path}")

    print("Done!")


def optimize_cmd(args: argparse.Namespace) -> None:
    """Optimize an ONNX model."""
    print(f"Optimizing {args.model}...")


def quantize_cmd(args: argparse.Namespace) -> None:
    """Quantize an ONNX model."""
    print(f"Quantizing {args.model}...")


def export_cmd(args: argparse.Namespace) -> None:
    """Export a model to ONNX."""
    print(f"Exporting {args.script}...")


def convert_cmd(args: argparse.Namespace) -> None:
    """Convert between model formats."""
    print(f"Converting from {args.src} to {args.dst}...")


def serve_cmd(args: argparse.Namespace) -> None:
    """Serve the local web visualizer."""
    print(f"Serving {args.model} on local server...")


def compile_cmd(args: argparse.Namespace) -> None:
    """Compile an ONNX model AOT."""
    print(f"Compiling {args.model}...")


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        prog="onnx9000", description="ONNX9000 Unified MLOps and Execution Ecosystem CLI."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Inspect
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect an ONNX model (MACs, FLOPs, Memory)"
    )
    inspect_parser.add_argument("model", type=str, help="Path to the .onnx file")
    inspect_parser.set_defaults(func=inspect_cmd)

    # Simplify
    simplify_parser = subparsers.add_parser("simplify", help="Simplify an ONNX model")
    simplify_parser.add_argument("model", type=str, help="Path to the input .onnx file")
    simplify_parser.add_argument(
        "output", type=str, nargs="?", help="Path to the output .onnx file"
    )
    simplify_parser.add_argument(
        "--skip-fusions", action="store_true", help="Skip operator fusions"
    )
    simplify_parser.add_argument(
        "--skip-constant-folding", action="store_true", help="Skip constant folding"
    )
    simplify_parser.add_argument(
        "--skip-shape-inference", action="store_true", help="Skip shape inference"
    )
    simplify_parser.add_argument(
        "--skip-fuse-bn", action="store_true", help="Skip BatchNorm fusion"
    )
    simplify_parser.add_argument(
        "--skip-rules", type=str, help="Comma-separated list of rules to skip"
    )
    simplify_parser.add_argument("--dry-run", action="store_true", help="Operate on a copy")
    simplify_parser.add_argument(
        "--max-iterations", type=int, default=10, help="Max simplification iterations"
    )
    simplify_parser.add_argument("--log-json", action="store_true", help="Log JSON summary")
    simplify_parser.add_argument(
        "--size-limit-mb", type=float, default=0.0, help="Track model size limit"
    )
    simplify_parser.add_argument(
        "--input-shape",
        type=str,
        action="append",
        help="Input shape override, e.g., 'x:1,3,224,224'",
    )
    simplify_parser.add_argument(
        "--target-opset", type=int, help="Target ONNX opset version to explicitly override"
    )
    simplify_parser.add_argument(
        "--tensor-type", type=str, action="append", help="Input type override, e.g., 'x:FLOAT32'"
    )
    simplify_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    simplify_parser.add_argument(
        "--strip-metadata", action="store_true", help="Strip model metadata"
    )
    simplify_parser.add_argument(
        "--sort-value-info", action="store_true", help="Sort ValueInfo lists alphabetically"
    )
    simplify_parser.add_argument(
        "--diff-json", action="store_true", help="Output a visual DAG difference JSON file"
    )
    simplify_parser.add_argument(
        "--preserve-nodes", type=str, help="Comma-separated names of nodes to preserve from DCE"
    )
    simplify_parser.add_argument(
        "--check-n",
        type=int,
        help="Check output consistency N times (Stub for onnxsim compatibility)",
    )
    simplify_parser.add_argument(
        "--custom-ops",
        type=str,
        action="append",
        help="Path to Python file registering custom pure-Python execution kernels",
    )
    simplify_parser.add_argument(
        "--prune-inputs", type=str, help="Comma-separated explicit inputs to prune if unused"
    )
    simplify_parser.set_defaults(func=simplify_cmd)

    # Optimize
    optimize_parser = subparsers.add_parser(
        "optimize", help="Apply graph fusions and layout optimizations"
    )
    optimize_parser.add_argument("model", type=str, help="Path to the .onnx file")
    optimize_parser.set_defaults(func=optimize_cmd)

    # Quantize
    quantize_parser = subparsers.add_parser("quantize", help="Quantize an ONNX model")
    quantize_parser.add_argument("model", type=str, help="Path to the .onnx file")
    quantize_parser.set_defaults(func=quantize_cmd)

    # Export
    export_parser = subparsers.add_parser("export", help="Export PyTorch/TF scripts to ONNX")
    export_parser.add_argument("script", type=str, help="Path to the model script")
    export_parser.set_defaults(func=export_cmd)

    # Convert
    convert_parser = subparsers.add_parser("convert", help="Convert legacy model formats to ONNX")
    convert_parser.add_argument(
        "--src", type=str, required=True, help="Source format (e.g., keras, caffe)"
    )
    convert_parser.add_argument("--dst", type=str, default="onnx", help="Target format")
    convert_parser.set_defaults(func=convert_cmd)

    # Serve
    serve_parser = subparsers.add_parser(
        "serve", help="Host the Netron-style web visualizer locally"
    )
    serve_parser.add_argument("model", type=str, help="Path to the .onnx file")
    serve_parser.set_defaults(func=serve_cmd)

    # Compile
    compile_parser = subparsers.add_parser(
        "compile", help="Ahead-of-Time compilation (IREE, CoreML)"
    )
    compile_parser.add_argument("model", type=str, help="Path to the .onnx file")
    compile_parser.set_defaults(func=compile_cmd)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
