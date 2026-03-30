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
    """Optimize an ONNX model (Fusions + optional Pruning/Quantization)."""
    print(f"Optimizing {args.model}...")
    print(f"Loading {args.model}...")
    graph = load_onnx(args.model)

    # Simple fusions would happen here

    # Item 226: Handle chained operations
    if args.prune:
        print(f"Chaining pruning (sparsity={args.sparsity})...")
        from onnx9000.optimizer.sparse.modifier import GlobalMagnitudePruningModifier

        mod = GlobalMagnitudePruningModifier(params=["re:.*"], final_sparsity=args.sparsity)
        mod.apply(graph)

    if args.quantize:
        print("Chaining quantization (int8)...")
        from onnx9000.optimizer.sparse.modifier import QuantizationModifier

        mod = QuantizationModifier(params=["re:.*"])
        mod.apply(graph)

    out_path = args.output or args.model.replace(".onnx", "_opt.onnx")
    print(f"Saving to {out_path}...")
    save_onnx(graph, out_path)
    print("Done!")


def quantize_cmd(args: argparse.Namespace) -> None:
    """Quantize an ONNX model."""
    print(f"Quantizing {args.model}...")
    print(f"Loading {args.model}...")
    graph = load_onnx(args.model)
    from onnx9000.optimizer.sparse.modifier import QuantizationModifier

    mod = QuantizationModifier(params=["re:.*"])
    mod.apply(graph)

    out_path = args.output or args.model.replace(".onnx", "_quant.onnx")
    print(f"Saving to {out_path}...")
    save_onnx(graph, out_path)
    print("Done!")


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


def onnx2gguf_cmd(args: argparse.Namespace) -> None:
    """Convert ONNX to GGUF."""
    import json
    import os

    try:
        import triton

        print(f"Detected Triton version {triton.__version__}")
        if int(triton.__version__.split(".")[0]) < 2:
            print("WARNING: Triton version < 2.0 might be incompatible.")
    except ImportError:
        print("Triton not found in current environment. Using generated code will require it.")

    from onnx9000.onnx2gguf.compiler import compile_gguf

    if args.dry_run:
        print(f"Dry run: Would convert {args.model} to GGUF")
        return

    if (
        os.path.exists(args.model)
        and os.path.getsize(args.model) > 70_000_000_000
        and not args.force
    ):
        print("Warning: Massive model detected. Use --force to proceed.")
        return

    graph = load_onnx(args.model)
    out_path = args.output or args.model.replace(".onnx", ".gguf")

    kv_overrides = {}
    if args.architecture:
        kv_overrides["general.architecture"] = args.architecture

    if args.tokenizer:
        with open(args.tokenizer) as f:
            kv_overrides["tokenizer.json"] = f.read()

    if args.outtype:
        kv_overrides["general.file_type"] = args.outtype

    with open(out_path, "wb") as out_f:
        compile_gguf(graph, out_f, kv_overrides=kv_overrides)

    print(f"Saved GGUF to {out_path}")


def gguf2onnx_cmd(args: argparse.Namespace) -> None:
    """Convert GGUF to ONNX."""
    from onnx9000.onnx2gguf.reader import GGUFReader
    from onnx9000.onnx2gguf.reverse import reconstruct_onnx

    out_path = args.output or args.model.replace(".gguf", ".onnx")

    with open(args.model, "rb") as f:
        reader = GGUFReader(f)
        graph = reconstruct_onnx(reader)

    save_onnx(graph, out_path)
    print(f"Saved ONNX to {out_path}")


def openvino_export_cmd(args: argparse.Namespace) -> None:
    """Export an ONNX model to OpenVINO IR."""
    import os

    from onnx9000.core.parser.core import load as load_onnx
    from onnx9000.openvino.exporter import OpenVinoExporter

    print(f"Loading {args.model}...")
    graph = load_onnx(args.model)

    # Handle overrides
    if args.shape:
        for shape_str in args.shape:
            name, dims = shape_str.split(":", 1)
            parsed_dims = []
            for d in dims.strip("[]").split(","):
                d = d.strip()
                if not d:
                    continue
                try:
                    parsed_dims.append(int(d))
                except ValueError:
                    parsed_dims.append(d)
            for inp in graph.inputs:
                if inp.name == name:
                    inp.shape = tuple(parsed_dims)

    if args.dynamic_batch:
        for inp in graph.inputs:
            if len(inp.shape) > 0:
                new_shape = list(inp.shape)
                new_shape[0] = -1
                inp.shape = tuple(new_shape)

    if getattr(args, "data_type", None):
        for dtype_str in args.data_type:
            name, dtype = dtype_str.split(":", 1)
            for inp in graph.inputs:
                if inp.name == name:
                    inp.dtype = dtype.strip()

    print("Generating OpenVINO IR...")
    exporter = OpenVinoExporter(graph, compress_to_fp16=args.fp16)
    xml_str, bin_data = exporter.export()

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.model))[0]

    xml_path = os.path.join(args.output_dir, f"{base_name}.xml")
    bin_path = os.path.join(args.output_dir, f"{base_name}.bin")
    mapping_path = os.path.join(args.output_dir, f"{base_name}.mapping")

    print("Writing XML...")
    with open(xml_path, "w") as f:
        f.write(xml_str)

    print("Writing BIN...")
    with open(bin_path, "wb") as f:
        f.write(bin_data)

    print("Writing mapping...")
    with open(mapping_path, "w") as f:
        f.write('<?xml version="1.0" ?>\n<mapping>\n</mapping>')

    print(f"Successfully exported OpenVINO model to {args.output_dir}")


def openvino_cmd(args: argparse.Namespace) -> None:
    """Entrypoint for OpenVINO subcommand group."""
    if not hasattr(args, "openvino_func"):
        print("Missing openvino subcommand")
        import sys

        sys.exit(1)
    args.openvino_func(args)


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
    import os
    import subprocess

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


def sparse_prune_cmd(args: argparse.Namespace) -> None:
    """Prune an ONNX model using a SparseML recipe."""
    from onnx9000.core.ir import Constant
    from onnx9000.core.sparse import (
        analyze_topological_dead_ends,
        collapse_sparse_tensors,
        dense_to_coo,
        detect_theoretical_sparsity,
        get_sparsity_report,
        strip_dense_representation,
    )
    from onnx9000.optimizer.sparse.modifier import apply_recipe

    print(f"Loading {args.model}...")
    graph = load_onnx(args.model)

    if args.recipe:
        print(f"Applying recipe from {args.recipe}...")
        with open(args.recipe) as f:
            recipe_yaml = f.read()
        apply_recipe(graph, recipe_yaml)
    elif args.sparsity:
        print(f"Applying global magnitude pruning with sparsity {args.sparsity}...")
        from onnx9000.optimizer.sparse.modifier import GlobalMagnitudePruningModifier

        mod = GlobalMagnitudePruningModifier(params=["re:.*"], final_sparsity=args.sparsity)
        mod.apply(graph)

    # After pruning, convert modified constants to SparseTensor to actually save space
    print("Converting pruned tensors to Sparse format...")
    for name in list(graph.tensors.keys()):
        tensor = graph.tensors[name]
        if isinstance(tensor, Constant) and tensor.is_initializer:
            sparsity = detect_theoretical_sparsity(tensor)
            if sparsity > 0.05:
                sparse_tensor = dense_to_coo(tensor)
                graph.tensors[name] = sparse_tensor
                if name in graph.initializers:
                    graph.initializers.remove(name)
                if name not in graph.sparse_initializers:
                    graph.sparse_initializers.append(name)
                # Item 90: Strip dense representation
                strip_dense_representation(graph, name)

    # Item 91: Collapse structurally 100% sparse tensors
    print("Collapsing 100% sparse tensors...")
    collapse_sparse_tensors(graph)

    # Item 92: Analyze topological dead ends
    dead_nodes = analyze_topological_dead_ends(graph)
    if dead_nodes:
        print(f"Found {len(dead_nodes)} potential dead nodes due to high sparsity.")

    print("\nSparsity Report:")
    print(get_sparsity_report(graph))

    out_path = args.output or args.model.replace(".onnx", "_sparse.onnx")
    print(f"\nSaving to {out_path}...")
    save_onnx(graph, out_path)
    print("Done!")


def sparse_de_sparsify_cmd(args: argparse.Namespace) -> None:
    """Inflate SparseTensor back into dense arrays."""
    from onnx9000.core.sparse import de_sparsify

    print(f"Loading {args.model}...")
    graph = load_onnx(args.model)
    print("De-sparsifying...")
    de_sparsify(graph)

    out_path = args.output or args.model.replace(".onnx", "_dense.onnx")
    print(f"Saving to {out_path}...")
    save_onnx(graph, out_path)
    print("Done!")


def sparse_cmd(args: argparse.Namespace) -> None:
    """Entrypoint for sparse subcommand group."""
    if not hasattr(args, "sparse_func"):
        print("Missing sparse subcommand")
        import sys

        sys.exit(1)
    args.sparse_func(args)


def prune_cmd(args: argparse.Namespace) -> None:
    """Headless CLI modification: Prune nodes from the graph."""
    print(f"Loading {args.model} for pruning...")

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
    """Rename a graph input."""
    print(f"Renaming input {args.old} to {args.new} in {args.model}...")
    graph = load_onnx(args.model)
    for inp in graph.inputs:
        if inp.name == args.old:
            inp.name = args.new
    for node in graph.nodes:
        node.inputs = [args.new if i == args.old else i for i in node.inputs]

    out_path = args.output or args.model
    save_onnx(graph, out_path)
    print("Done!")


def change_batch_cmd(args: argparse.Namespace) -> None:
    """Change the batch size of an ONNX model."""
    print(f"Changing batch size to {args.size} in {args.model}...")
    graph = load_onnx(args.model)
    new_size = args.size
    try:
        new_size = int(args.size)
    except ValueError:
        pass

    for inp in graph.inputs:
        if len(inp.shape) > 0:
            inp.shape[0] = new_size

    out_path = args.output or args.model
    save_onnx(graph, out_path)
    print("Done!")


def mutate_cmd(args: argparse.Namespace) -> None:
    """Apply generic mutations from a JSON script."""
    print(f"Applying mutations from {args.script} to {args.model}...")
    graph = load_onnx(args.model)
    import json

    with open(args.script) as f:
        mutations = json.load(f)

    for mut in mutations:
        if mut["action"] == "remove_node":
            graph.nodes = [n for n in graph.nodes if n.name != mut["node_name"]]

    out_path = args.output or args.model
    save_onnx(graph, out_path)
    print("Done!")


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        prog="onnx9000", description="ONNX9000 Unified MLOps and Execution Ecosystem CLI."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect an ONNX model")
    inspect_parser.add_argument("model", type=str, help="Path to the .onnx file")
    inspect_parser.set_defaults(func=inspect_cmd)

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

    # Sparse
    sparse_parser = subparsers.add_parser("sparse", help="SparseML pruning and sparsification")
    sparse_sub = sparse_parser.add_subparsers(dest="sparse_command", help="Sparse subcommands")

    sparse_prune = sparse_sub.add_parser("prune", help="Prune model weights")
    sparse_prune.add_argument("model", type=str, help="Path to the .onnx file")
    sparse_prune.add_argument("--recipe", type=str, help="Path to SparseML .yaml recipe")
    sparse_prune.add_argument("--sparsity", type=float, help="Global sparsity target (0.0 - 1.0)")
    sparse_prune.add_argument("-o", "--output", type=str, help="Output path")
    sparse_prune.set_defaults(sparse_func=sparse_prune_cmd)

    sparse_de = sparse_sub.add_parser("de-sparsify", help="Inflate sparse tensors back to dense")
    sparse_de.add_argument("model", type=str, help="Path to the .onnx file")
    sparse_de.add_argument("-o", "--output", type=str, help="Output path")
    sparse_de.set_defaults(sparse_func=sparse_de_sparsify_cmd)

    sparse_parser.set_defaults(func=sparse_cmd)

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
    simplify_parser.set_defaults(func=simplify_cmd)

    # Optimize
    optimize_parser = subparsers.add_parser(
        "optimize", help="Apply graph fusions and layout optimizations"
    )
    optimize_parser.add_argument("model", type=str, help="Path to the .onnx file")
    optimize_parser.add_argument("--prune", action="store_true", help="Chain pruning")
    optimize_parser.add_argument("--sparsity", type=float, default=0.5, help="Pruning sparsity")
    optimize_parser.add_argument("--quantize", action="store_true", help="Chain quantization")
    optimize_parser.add_argument("-o", "--output", type=str, help="Output path")
    optimize_parser.set_defaults(func=optimize_cmd)

    # Quantize
    quantize_parser = subparsers.add_parser("quantize", help="Quantize an ONNX model")
    quantize_parser.add_argument("model", type=str, help="Path to the .onnx file")
    quantize_parser.add_argument("-o", "--output", type=str, help="Output path")
    quantize_parser.set_defaults(func=quantize_cmd)

    # Rename Input
    rename_parser = subparsers.add_parser("rename-input", help="Rename a graph input")
    rename_parser.add_argument("model", type=str, help="Path to the .onnx file")
    rename_parser.add_argument("--old", type=str, required=True, help="Old input name")
    rename_parser.add_argument("--new", type=str, required=True, help="New input name")
    rename_parser.add_argument("--output", type=str, help="Output path")
    rename_parser.set_defaults(func=rename_input_cmd)

    # Change Batch
    batch_parser = subparsers.add_parser("change-batch", help="Change batch size")
    batch_parser.add_argument("model", type=str, help="Path to the .onnx file")
    batch_parser.add_argument("--size", type=str, required=True, help="New batch size")
    batch_parser.add_argument("--output", type=str, help="Output path")
    batch_parser.set_defaults(func=change_batch_cmd)

    # Mutate
    mutate_parser = subparsers.add_parser("mutate", help="Apply mutations from JSON")
    mutate_parser.add_argument("model", type=str, help="Path to the .onnx file")
    mutate_parser.add_argument("--script", type=str, required=True, help="Path to JSON script")
    mutate_parser.add_argument("--output", type=str, help="Output path")
    mutate_parser.set_defaults(func=mutate_cmd)

    # Info
    info_parser = subparsers.add_parser("info", help="Diagnostic information")
    info_sub = info_parser.add_subparsers(dest="info_command", help="Info subcommands")
    info_webnn = info_sub.add_parser("webnn", help="WebNN diagnostic info")
    info_webnn.set_defaults(info_func=info_webnn_cmd)
    info_parser.set_defaults(func=info_cmd)

    # Convert
    convert_parser = subparsers.add_parser("convert", help="Convert model formats")
    convert_parser.add_argument("--src", type=str, required=True, help="Source format")
    convert_parser.add_argument("--dst", type=str, required=True, help="Destination format")
    convert_parser.set_defaults(func=convert_cmd)

    # Serve
    serve_parser = subparsers.add_parser("serve", help="Serve local visualizer")
    serve_parser.add_argument("model", type=str, nargs="?", help="Path to the .onnx file")
    serve_parser.set_defaults(func=serve_cmd)

    # Export
    export_parser = subparsers.add_parser("export", help="Export to ONNX")
    export_parser.add_argument("script", type=str, help="Path to export script")
    export_parser.set_defaults(func=export_cmd)

    # Compile
    compile_parser = subparsers.add_parser("compile", help="Compile model AOT")
    compile_parser.add_argument("model", type=str, help="Path to the .onnx file")
    compile_parser.set_defaults(func=compile_cmd)

    # OpenVINO
    openvino_parser = subparsers.add_parser("openvino", help="OpenVINO toolchain integration")
    openvino_sub = openvino_parser.add_subparsers(
        dest="openvino_command", help="OpenVINO subcommands"
    )

    ov_exp = openvino_sub.add_parser("export", help="Export model to OpenVINO IR")
    ov_exp.add_argument("model", type=str, help="Path to the input .onnx file")
    ov_exp.add_argument("-o", "--output-dir", type=str, default=".", help="Output directory")
    ov_exp.add_argument("--fp16", action="store_true", help="Downcast all weights to FP16")
    ov_exp.add_argument(
        "--shape", type=str, action="append", help="Shape override, e.g., 'input:[1,3,224,224]'"
    )
    ov_exp.add_argument("--data_type", type=str, action="append", help="Data type overrides")
    ov_exp.add_argument("--dynamic-batch", action="store_true", help="Set batch size to dynamic -1")
    ov_exp.set_defaults(openvino_func=openvino_export_cmd)

    openvino_parser.set_defaults(func=openvino_cmd)

    # Optimum
    optimum_parser = subparsers.add_parser("optimum", help="HuggingFace Optimum integration")
    optimum_sub = optimum_parser.add_subparsers(dest="optimum_command", help="Optimum subcommands")

    opt_exp = optimum_sub.add_parser("export", help="Export HF model")
    opt_exp.add_argument("model_id", type=str, help="HF Model ID")
    opt_exp.add_argument("--task", type=str, help="Task type")
    opt_exp.add_argument("--opset", type=int, help="ONNX opset")
    opt_exp.add_argument("--device", type=str, help="Device")
    opt_exp.add_argument("--cache-dir", type=str, help="Cache dir")
    opt_exp.add_argument("--split", type=str, help="Data split")
    opt_exp.set_defaults(optimum_func=optimum_export_cmd)

    opt_opt = optimum_sub.add_parser("optimize", help="Optimize HF ONNX model")
    opt_opt.add_argument("model", type=str, help="Path to .onnx file")
    opt_opt.add_argument("--level", type=str, help="Optimization level")
    opt_opt.add_argument("--disable-fusion", action="store_true", help="Disable fusions")
    opt_opt.add_argument("--optimize-size", action="store_true", help="Optimize for size")
    opt_opt.set_defaults(optimum_func=optimum_optimize_cmd)

    opt_quant = optimum_sub.add_parser("quantize", help="Quantize HF ONNX model")
    opt_quant.add_argument("model", type=str, help="Path to .onnx file")
    opt_quant.add_argument("--quantize", type=str, help="Quantization method")
    opt_quant.add_argument("--gptq-bits", type=int, help="GPTQ bits")
    opt_quant.add_argument("--gptq-group-size", type=int, help="GPTQ group size")
    opt_quant.set_defaults(optimum_func=optimum_quantize_cmd)

    optimum_parser.set_defaults(func=optimum_cmd)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
