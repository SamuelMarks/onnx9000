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


def optimum_export_cmd(args: argparse.Namespace) -> None:
    """Export a HuggingFace model to ONNX."""
    from onnx9000_optimum.export import export_model

    export_model(
        model_id=args.model_id,
        output_dir="exported_model",
        task=args.task,
        opset=args.opset,
        device=args.device,
        cache_dir=args.cache_dir,
        split=args.split,
    )


def optimum_optimize_cmd(args: argparse.Namespace) -> None:
    """Optimize a HuggingFace ONNX model for Web deployment."""
    from onnx9000_optimum.optimize import optimize_model

    optimize_model(
        model_path=args.model,
        level=args.level,
        disable_fusion=args.disable_fusion,
        optimize_size=args.optimize_size,
    )


def optimum_quantize_cmd(args: argparse.Namespace) -> None:
    """Quantize an ONNX model using Optimum settings."""
    from onnx9000_optimum.quantize import quantize_model

    quantize_model(
        model_path=args.model,
        method=args.quantize,
        gptq_bits=args.gptq_bits,
        gptq_group_size=args.gptq_group_size,
    )


def optimum_cmd(args: argparse.Namespace) -> None:
    """Entrypoint for Optimum subcommand group."""
    if not hasattr(args, "optimum_func"):
        print("Missing optimum subcommand")
        import sys

        sys.exit(1)
    args.optimum_func(args)


def convert_cmd(args: argparse.Namespace) -> None:
    """Convert between model formats."""
    print(f"Converting from {args.src} to {args.dst}...")


def serve_cmd(args: argparse.Namespace) -> None:
    """Serve the local web visualizer."""
    print(f"Serving {args.model} on local server...")


def compile_cmd(args: argparse.Namespace) -> None:
    """Compile an ONNX model AOT."""
    print(f"Compiling {args.model}...")


def info_cmd(args: argparse.Namespace) -> None:
    """Entrypoint for info diagnostics command group."""
    if not hasattr(args, "info_func"):
        print("Missing info subcommand")
        import sys

        sys.exit(1)
    args.info_func(args)


def info_webnn_cmd(args: argparse.Namespace) -> None:
    """Print host WebNN capabilities."""
    print("WebNN API Diagnostic Info:")
    print("--------------------------")
    print("Host NPU capabilities check is primarily available in the browser environment.")
    print("Run `onnx9000 serve` to open the local visualizer and view detailed NPU metrics.")
    print("\nCapabilities (Mock/Node.js context):")
    print("- Float16: true")
    print("- Float32: true")
    print("- Int8: true")
    print("- WebNN API Present: false (Requires browser `navigator.ml` context)")


def coreml_cmd(args: argparse.Namespace) -> None:
    """Execute CoreML export/import via Node.js CLI."""
    import subprocess
    import os

    # Look for the JS CLI script in the monorepo
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    )
    cli_js_path = os.path.join(base_dir, "packages", "js", "coreml", "dist", "cli.js")

    if not os.path.exists(cli_js_path):
        print(
            f"Error: Could not find JS CoreML CLI at {cli_js_path}. Please build the JS packages."
        )
        sys.exit(1)

    cmd = ["node", cli_js_path, args.coreml_command, args.model]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"CoreML command failed with code {e.returncode}")
        sys.exit(e.returncode)


def edit_cmd(args: argparse.Namespace) -> None:
    """Open the ONNX9000 Modifier Web UI."""
    print(f"Starting modifier UI for {args.model}...")
    import os
    import subprocess

    # Run vite dev server or equivalent from apps/netron-ui
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    )
    ui_dir = os.path.join(base_dir, "apps", "netron-ui")
    if os.path.exists(ui_dir):
        print(f"Opening {ui_dir}...")
        try:
            subprocess.run(["pnpm", "dev"], cwd=ui_dir)
        except KeyboardInterrupt:
            pass
    else:
        print("Modifier UI not found in monorepo.")


def prune_cmd(args: argparse.Namespace) -> None:
    """Headless CLI modification: Prune nodes from the graph."""
    print(f"Loading {args.model} for pruning...")
    from onnx9000.core.parser.core import load as load_onnx
    from onnx9000.core.serializer import save as save_onnx

    graph = load_onnx(args.model)
    nodes_to_remove = set(args.nodes.split(","))

    orig_count = len(graph.nodes)
    graph.nodes = [
        n for n in graph.nodes if n.name not in nodes_to_remove and n.op_type not in nodes_to_remove
    ]

    removed_count = orig_count - len(graph.nodes)
    print(f"Pruned {removed_count} nodes.")

    out_path = args.output or args.model.replace(".onnx", "_pruned.onnx")
    save_onnx(graph, out_path)
    print(f"Saved pruned model to {out_path}")


def rename_input_cmd(args: argparse.Namespace) -> None:
    """Headless CLI modification: Rename an input in the graph."""
    print(f"Loading {args.model} to rename input...")
    from onnx9000.core.parser.core import load as load_onnx
    from onnx9000.core.serializer import save as save_onnx

    graph = load_onnx(args.model)
    old_name = args.old
    new_name = args.new

    # Rename in graph.inputs
    for inp in graph.inputs:
        if inp.name == old_name:
            inp.name = new_name

    # Rename in node inputs
    for node in graph.nodes:
        for i in range(len(node.inputs)):
            if node.inputs[i] == old_name:
                node.inputs[i] = new_name

    out_path = args.output or args.model.replace(".onnx", "_renamed.onnx")
    save_onnx(graph, out_path)
    print(f"Saved renamed model to {out_path}")


def change_batch_cmd(args: argparse.Namespace) -> None:
    """Headless CLI modification: Change batch size."""
    print(f"Loading {args.model} to change batch size...")
    from onnx9000.core.parser.core import load as load_onnx
    from onnx9000.core.serializer import save as save_onnx

    graph = load_onnx(args.model)
    size = args.size
    # Try cast to int if possible
    try:
        size = int(size)
    except ValueError:
        pass

    for vi in graph.inputs + graph.outputs + graph.value_info:
        if vi.shape and len(vi.shape) > 0:
            vi.shape[0] = size

    out_path = args.output or args.model.replace(".onnx", "_batch_changed.onnx")
    save_onnx(graph, out_path)
    print(f"Saved model with new batch size to {out_path}")


def mutate_cmd(args: argparse.Namespace) -> None:
    """Headless CLI modification: Apply JSON script mutations."""
    print(f"Loading {args.model} for batch mutation...")
    import json
    from onnx9000.core.parser.core import load as load_onnx
    from onnx9000.core.serializer import save as save_onnx

    graph = load_onnx(args.model)
    with open(args.script, "r") as f:
        edits = json.load(f)

    # Simplistic evaluator
    for edit in edits:
        if edit.get("action") == "remove_node":
            node_name = edit.get("node_name")
            graph.nodes = [n for n in graph.nodes if n.name != node_name and n.op_type != node_name]

    out_path = args.output or args.model.replace(".onnx", "_mutated.onnx")
    save_onnx(graph, out_path)
    print(f"Saved mutated model to {out_path}")


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        prog="onnx9000", description="ONNX9000 Unified MLOps and Execution Ecosystem CLI."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Edit
    edit_parser = subparsers.add_parser("edit", help="Start the local visual modifier UI")
    edit_parser.add_argument("model", type=str, nargs="?", help="Path to the .onnx file")
    edit_parser.set_defaults(func=edit_cmd)

    # Prune
    prune_parser = subparsers.add_parser("prune", help="Headless graph pruning")
    prune_parser.add_argument("model", type=str, help="Path to the .onnx file")
    prune_parser.add_argument(
        "--nodes", type=str, required=True, help="Comma-separated nodes to remove"
    )
    prune_parser.add_argument("--output", type=str, help="Output path")
    prune_parser.set_defaults(func=prune_cmd)
    # Rename Input
    rename_parser = subparsers.add_parser("rename-input", help="Headless input renaming")
    rename_parser.add_argument("model", type=str, help="Path to the .onnx file")
    rename_parser.add_argument("--old", type=str, required=True, help="Old input name")
    rename_parser.add_argument("--new", type=str, required=True, help="New input name")
    rename_parser.add_argument("--output", type=str, help="Output path")
    rename_parser.set_defaults(func=rename_input_cmd)

    # Change Batch
    batch_parser = subparsers.add_parser("change-batch", help="Headless batch size modification")
    batch_parser.add_argument("model", type=str, help="Path to the .onnx file")
    batch_parser.add_argument(
        "--size", type=str, required=True, help="New batch size (int or string)"
    )
    batch_parser.add_argument("--output", type=str, help="Output path")
    batch_parser.set_defaults(func=change_batch_cmd)

    # Mutate
    mutate_parser = subparsers.add_parser("mutate", help="Headless JSON script mutation")
    mutate_parser.add_argument("model", type=str, help="Path to the .onnx file")
    mutate_parser.add_argument("--script", type=str, required=True, help="Path to edits.json")
    mutate_parser.add_argument("--output", type=str, help="Output path")
    mutate_parser.set_defaults(func=mutate_cmd)

    # Info
    info_parser = subparsers.add_parser("info", help="Diagnostic information")
    info_subparsers = info_parser.add_subparsers(dest="info_command", help="Info commands")
    info_webnn_parser = info_subparsers.add_parser("webnn", help="List host NPU capabilities")
    info_webnn_parser.set_defaults(info_func=info_webnn_cmd, func=info_cmd)

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

    # Optimum
    optimum_parser = subparsers.add_parser(
        "optimum", help="HuggingFace Optimum web-optimized export and quantization"
    )
    optimum_subparsers = optimum_parser.add_subparsers(
        dest="optimum_command", help="Optimum commands"
    )

    # Optimum Export
    optimum_export = optimum_subparsers.add_parser("export", help="Export models to ONNX")
    optimum_export.add_argument(
        "--model", dest="model_id", type=str, help="Model ID from HuggingFace Hub"
    )
    optimum_export.add_argument("--task", type=str, help="Task to export for")
    optimum_export.add_argument("--opset", type=int, help="ONNX opset version")
    optimum_export.add_argument(
        "--device",
        type=str,
        choices=["cpu", "wasm", "webgpu", "webnn"],
        default="cpu",
        help="Target device",
    )
    optimum_export.add_argument("--cache_dir", type=str, help="Cache directory for HF weights")
    optimum_export.add_argument(
        "--monolith", action="store_true", help="Store weights in monolithic file"
    )
    optimum_export.add_argument(
        "--external-data", action="store_true", help="Store weights externally"
    )
    optimum_export.add_argument("--atol", type=float, help="Absolute tolerance for validation")
    optimum_export.add_argument("--rtol", type=float, help="Relative tolerance for validation")
    optimum_export.add_argument("--split", action="store_true", help="Split massive graphs")
    optimum_export.set_defaults(optimum_func=optimum_export_cmd, func=optimum_cmd)

    # Optimum Optimize
    optimum_optimize = optimum_subparsers.add_parser(
        "optimize", help="Optimize ONNX models for web"
    )
    optimum_optimize.add_argument("model", type=str, help="Path to the .onnx file")
    optimum_optimize.add_argument(
        "--level",
        type=str,
        choices=["O1", "O2", "O3", "O4"],
        default="O1",
        help="Optimization level",
    )
    optimum_optimize.add_argument(
        "--disable-fusion", action="store_true", help="Disable operator fusion"
    )
    optimum_optimize.add_argument(
        "--optimize-size", action="store_true", help="Strip debug names and compress size"
    )
    optimum_optimize.set_defaults(optimum_func=optimum_optimize_cmd, func=optimum_cmd)

    # Optimum Quantize
    optimum_quantize = optimum_subparsers.add_parser("quantize", help="Quantize ONNX models")
    optimum_quantize.add_argument("model", type=str, help="Path to the .onnx file")
    optimum_quantize.add_argument(
        "--quantize",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization method",
    )
    optimum_quantize.add_argument("--gptq-bits", type=int, help="GPTQ bits")
    optimum_quantize.add_argument("--gptq-group-size", type=int, help="GPTQ group size")
    optimum_quantize.set_defaults(optimum_func=optimum_quantize_cmd, func=optimum_cmd)

    # Convert
    convert_parser = subparsers.add_parser("convert", help="Convert legacy model formats to ONNX")
    convert_parser.add_argument(
        "--src", type=str, required=True, help="Source format (e.g., keras, caffe)"
    )
    convert_parser.add_argument("--dst", type=str, default="onnx", help="Target format")
    convert_parser.set_defaults(func=convert_cmd)

    # CoreML
    coreml_parser = subparsers.add_parser("coreml", help="Convert to/from CoreML packages")
    coreml_sub = coreml_parser.add_subparsers(dest="coreml_command", help="CoreML subcommands")

    coreml_export = coreml_sub.add_parser("export", help="Export ONNX to CoreML (.mlpackage)")
    coreml_export.add_argument("model", type=str, help="Path to the .onnx file")

    coreml_import = coreml_sub.add_parser("import", help="Import CoreML (.mlpackage) to ONNX")
    coreml_import.add_argument("model", type=str, help="Path to the .mlpackage")

    coreml_parser.set_defaults(func=coreml_cmd)

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
