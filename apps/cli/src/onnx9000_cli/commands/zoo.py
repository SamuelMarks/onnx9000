"""Model zoo, progressive loading, and workspace CLI commands."""

import argparse
import sys


def zoo_cmd(args: argparse.Namespace) -> None:
    """Entrypoint for Zoo model downloads and streaming."""
    print(f"Executing Zoo command: {args.zoo_command}")

    try:
        from onnx9000.zoo.catalog import ModelCatalog
        from onnx9000.zoo.tensors import BFloat16Upcaster, SafeTensorsMmapParser

        _ = (ModelCatalog, BFloat16Upcaster)
        _ = (ModelCatalog, BFloat16Upcaster)

        print("Zoo subsystem loaded.")
        if args.zoo_command == "download":
            print(f"Downloading {args.model_name}...")
        elif args.zoo_command == "inspect-safetensors":
            parser = SafeTensorsMmapParser(args.model_name)
            parser.parse()
            print(f"Safetensors header: {parser.header}")
    except ImportError as e:
        print(f"Zoo subsystem not fully initialized: {e}")
        sys.exit(1)


def progressive_loading_cmd(args: argparse.Namespace) -> None:
    """Generate progressive loading chunks."""
    try:
        from onnx9000_progressive_loading import ProgressiveLoading

        print(f"Generating progressive loading chunks for {args.model}...")
        obj = ProgressiveLoading()
        print(obj.process(args.model))  # pragma: no cover
        print("Success.")
    except ImportError:  # pragma: no cover
        print("onnx9000-progressive-loading not installed")


def new_model_arch_cmd(args: argparse.Namespace) -> None:
    """Scaffold a new model architecture."""
    try:
        from onnx9000_new_model_arch import NewModelArch

        print(f"Scaffolding new model architecture for: {args.arch}...")
        obj = NewModelArch()
        print(obj.process(args.arch))
        print("Success.")
    except ImportError:  # pragma: no cover
        print("onnx9000-new-model-arch not installed")


def workspace_cmd(args: argparse.Namespace) -> None:
    """Initialize a workspace."""
    try:
        from onnx9000_workspace import setup_workspace
    except ImportError:  # pragma: no cover
        import os
        import sys

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        sys.path.insert(0, base_dir)
        from onnx9000_workspace import setup_workspace

    setup_workspace(args.path)
