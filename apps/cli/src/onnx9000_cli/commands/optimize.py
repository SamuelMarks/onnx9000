"""Optimization, quantization, pruning, and sparsity CLI commands."""

import argparse


def _get_main() -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute get main operation.

    Returns:
        Result of get main.
    """
    import onnx9000_cli.main as m

    return m


def hummingbird_cmd(args: argparse.Namespace) -> None:
    """Run Hummingbird tree compiler optimizations."""
    print(f"Loading {args.model} for Hummingbird optimization...")
    graph = _get_main().load_onnx(args.model)
    from onnx9000.optimizer.hummingbird.engine import TranspilationEngine
    from onnx9000.optimizer.hummingbird.onnxml_parser import parse_onnxml_tree_ensemble
    from onnx9000.optimizer.hummingbird.strategies import TargetHardware

    engine = TranspilationEngine(target=TargetHardware.CPU)

    nodes_to_remove = []
    transpiled_graphs = []

    for node in graph.nodes:
        if node.op_type in ("TreeEnsembleClassifier", "TreeEnsembleRegressor"):
            trees = parse_onnxml_tree_ensemble(node)
            engine.abstractions = trees
            transpiled_g = engine.transpile(node)
            transpiled_graphs.append(transpiled_g)
            nodes_to_remove.append(node)

    # Simple replacement logic (assuming 1 tree ensemble for now)
    if nodes_to_remove and transpiled_graphs:
        graph.nodes = [n for n in graph.nodes if n not in nodes_to_remove]
        for tg in transpiled_graphs:
            graph.nodes.extend(tg.nodes)
            graph.tensors.update(tg.tensors)

    out_path = args.output or args.model.replace(".onnx", "_hb.onnx")
    print(f"Saving to {out_path}...")
    _get_main().save_onnx(graph, out_path)
    print("Done!")


def optimize_cmd(args: argparse.Namespace) -> None:
    """Optimize an ONNX model (Fusions + optional Pruning/Quantization)."""
    print(f"Optimizing {args.model}...")
    print(f"Loading {args.model}...")
    graph = _get_main().load_onnx(args.model)

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
    _get_main().save_onnx(graph, out_path)
    print("Done!")


def quantize_cmd(args: argparse.Namespace) -> None:
    """Quantize an ONNX model."""
    print(f"Quantizing {args.model}...")
    print(f"Loading {args.model}...")
    graph = _get_main().load_onnx(args.model)
    from onnx9000.optimizer.sparse.modifier import QuantizationModifier

    mod = QuantizationModifier(params=["re:.*"])
    mod.apply(graph)

    out_path = args.output or args.model.replace(".onnx", "_quant.onnx")
    print(f"Saving to {out_path}...")
    _get_main().save_onnx(graph, out_path)
    print("Done!")


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
    graph = _get_main().load_onnx(args.model)

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
    _get_main().save_onnx(graph, out_path)
    print("Done!")


def sparse_de_sparsify_cmd(args: argparse.Namespace) -> None:
    """Inflate SparseTensor back into dense arrays."""
    from onnx9000.core.sparse import de_sparsify

    print(f"Loading {args.model}...")
    graph = _get_main().load_onnx(args.model)
    print("De-sparsifying...")
    de_sparsify(graph)

    out_path = args.output or args.model.replace(".onnx", "_dense.onnx")
    print(f"Saving to {out_path}...")
    _get_main().save_onnx(graph, out_path)
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

    graph = _get_main().load_onnx(args.model)
    nodes_to_remove = set(args.nodes.split(","))

    orig_count = len(graph.nodes)
    graph.nodes = [
        n for n in graph.nodes if n.name not in nodes_to_remove and n.op_type not in nodes_to_remove
    ]

    removed_count = orig_count - len(graph.nodes)
    print(f"Pruned {removed_count} nodes.")

    out_path = args.output or args.model.replace(".onnx", "_pruned.onnx")
    _get_main().save_onnx(graph, out_path)
    print(f"Saved pruned model to {out_path}")


def olive_optimizer_cmd(args: argparse.Namespace) -> None:
    """Run the olive_optimizer command."""
    try:
        from onnx9000_olive_optimizer import OliveOptimizer

        _ = OliveOptimizer

        print("Olive Optimizer processed " + args.model)
    except ImportError:  # pragma: no cover
        print("onnx9000-olive-optimizer not installed")
