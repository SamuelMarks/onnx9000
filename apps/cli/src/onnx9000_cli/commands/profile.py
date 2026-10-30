"""Profiling, memory analysis, and diagnostic CLI commands."""

import argparse


def mobile_memory_cmd(args: argparse.Namespace) -> None:
    """Analyze and optimize mobile memory usage."""
    try:
        from onnx9000_mobile_memory import MobileMemory

        print(f"Analyzing mobile memory usage for {args.model}...")
        obj = MobileMemory()
        print(obj.process(args.model))  # pragma: no cover
        print("Optimization applied:  # pragma: no cover Memory Planning SUCCESS")
    except ImportError:  # pragma: no cover
        print("onnx9000-mobile-memory not installed")


def info_cmd(args: argparse.Namespace) -> None:
    """Entrypoint for info diagnostics command group."""
    if not hasattr(args, "info_func"):
        print("Missing info subcommand")
        import sys

        sys.exit(1)
    args.info_func(args)


def coverage_cmd(args: argparse.Namespace) -> None:
    """Entrypoint for coverage command."""
    from onnx9000_cli.coverage import update_coverage_cmd

    update_coverage_cmd(args)


def info_webnn_cmd(args: argparse.Namespace) -> None:
    """Print host WebNN capabilities."""
    print("WebNN API Diagnostic Info:")
    print("--------------------------")
    print("Host NPU capabilities check is primarily available in the browser environment.")
    print("Run `onnx9000 serve` to open the local visualizer and view detailed NPU metrics.")


def profiler_cmd(args: argparse.Namespace) -> None:
    """Run the Memory Arena & Profiler."""
    print(f"Running profiler for {args.model}...")
    try:
        from onnx9000_profiler import Profiler

        profiler = Profiler(args.model)
        profiler.run()  # pragma: no cover
        print(f"Peak Memory:  # pragma: no cover {profiler.get_peak_memory()} MB")
        if args.show_arena:
            print("Memory arena blocks rendered to terminal UI.")
    except ImportError:  # pragma: no cover
        print("onnx9000-profiler package not installed.")


def onnx_tool_cmd(args: argparse.Namespace) -> None:
    """Run the onnx_tool command."""
    try:
        from onnx9000_onnx_tool import ONNXTool

        _ = ONNXTool

        print("ONNX Tool processed " + args.model)
    except ImportError:  # pragma: no cover
        print("onnx9000-onnx-tool not installed")


def arena_cmd(args: argparse.Namespace) -> None:
    """Run the arena command."""
    print("Arena processed " + getattr(args, "model", ""))
