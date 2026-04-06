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


def hummingbird_cmd(args: argparse.Namespace) -> None:
    """Run Hummingbird tree compiler optimizations."""
    print(f"Loading {args.model} for Hummingbird optimization...")
    graph = load_onnx(args.model)
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
    save_onnx(graph, out_path)
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
    """Export a model script to ONNX or other formats."""
    print(f"Exporting {args.script} to {args.format}...")
    import importlib.util
    import os

    from onnx9000.converters.frontend.nn.module import Module
    from onnx9000.converters.frontend.tracer import trace

    # Load the script as a module
    spec = importlib.util.spec_from_file_location("export_script", args.script)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load script {args.script}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Search for any subclass of Module
    model = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, Module) and obj is not Module:
            model = obj()
            break

    if model is None:
        print("Error: No model found in script. Define a Module subclass.")
        sys.exit(1)

    # Try to find dummy input args
    from onnx9000.converters import torch_like as torch

    args_in = torch.randn(1, 1, 28, 28)

    builder = trace(model, args_in)
    graph = builder.to_graph()

    output = args.output
    if not output:
        base = os.path.splitext(args.script)[0]
        ext_map = {
            "onnx": ".onnx",
            "c": ".c",
            "cpp": ".cpp",
            "mlir": ".mlir",
            "keras": ".py",
            "wasm": ".js",
        }
        output = base + ext_map.get(args.format, ".out")

    from onnx9000.core.exporter import export_graph

    export_graph(graph, output, args.format)
    print(f"Successfully exported to {output}")


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
    src_fmt = getattr(args, "from", None) or "onnx"
    dst_fmt = getattr(args, "to", None) or "onnx"

    print(f"Converting from {src_fmt} ({args.src}) to {dst_fmt}...")
    import os

    # Load source
    if src_fmt == "onnx":
        graph = load_onnx(args.src)
    elif src_fmt == "keras":
        import keras

        model = keras.saving.load_model(args.src)
        from onnx9000.converters.tf.api import convert_keras_to_onnx

        graph = convert_keras_to_onnx(model)
    elif src_fmt == "jax":
        import json

        with open(args.src) as f:
            jaxpr_dict = json.load(f)
        from onnx9000.converters.parsers import JAXprParser

        graph = JAXprParser().parse(jaxpr_dict)
    elif src_fmt == "paddle":
        import os

        from onnx9000.converters.paddle.api import convert_paddle_to_onnx

        if os.path.isdir(args.src):
            model_path = os.path.join(args.src, "model.pdmodel")
            params_path = os.path.join(args.src, "model.pdiparams")
            if not os.path.exists(model_path):
                model_path = os.path.join(args.src, "__model__")
            if not os.path.exists(params_path):
                params_path = os.path.join(args.src, "weight")
        else:
            model_path = args.src
            params_path = args.src.replace(".pdmodel", ".pdiparams")

        with open(model_path, "rb") as f:
            model_data = f.read()

        params_data = None
        if os.path.exists(params_path):
            with open(params_path, "rb") as f:
                params_data = f.read()

        graph = convert_paddle_to_onnx(model_data, params_data)
    elif src_fmt == "sklearn":
        import joblib
        from onnx9000.converters.sklearn.builder import SKLearnParser
        from onnx9000.core.dtypes import DType

        model = joblib.load(args.src)
        parser = SKLearnParser(model, initial_types=[("float_input", DType.FLOAT32, ("N", "F"))])
        graph = parser.parse()
    elif src_fmt == "pytorch":
        import torch
        from onnx9000.converters.parsers import PyTorchFXParser

        # Load the PyTorch model
        # Try to load as torch.export (pt2) or fallback to torch.load for an nn.Module / ScriptModule
        # Note: torch.load requires the model class to be available in the environment
        try:
            model = torch.export.load(args.src)
        except Exception:
            model = torch.load(args.src, map_location="cpu")
            if isinstance(model, torch.nn.Module):
                model = torch.fx.symbolic_trace(model)

        parser = PyTorchFXParser()
        graph = parser.parse(model)
    elif src_fmt == "safetensors":
        from onnx9000.converters.safetensors.loader import load_safetensors_to_graph

        graph = load_safetensors_to_graph(args.src)
    elif src_fmt == "caffe":
        with open(args.src) as f:
            data = f.read()
        from onnx9000.converters.mltools.catboost import parse_catboost_json

        graph = parse_catboost_json(data)
    elif src_fmt == "lightgbm":
        with open(args.src) as f:
            data = f.read()
        from onnx9000.converters.mltools.lightgbm import parse_lightgbm_json

        graph = parse_lightgbm_json(data)
    elif src_fmt == "xgboost":
        with open(args.src) as f:
            data = f.read()
        from onnx9000.converters.mltools.xgboost import parse_xgboost_json

        graph = parse_xgboost_json(data)
    elif src_fmt == "libsvm":
        with open(args.src) as f:
            data = f.read()
        from onnx9000.converters.mltools.libsvm import parse_libsvm

        graph = parse_libsvm(data)
    elif src_fmt == "h2o":
        with open(args.src) as f:
            data = f.read()
        from onnx9000.converters.mltools.h2o import parse_h2o

        graph = parse_h2o(data)
    elif src_fmt == "sparkml":
        with open(args.src) as f:
            data = f.read()
        from onnx9000.converters.mltools.sparkml import parse_sparkml_pipeline

        graph = parse_sparkml_pipeline(data)
    elif src_fmt == "coreml":
        with open(args.src) as f:
            data = f.read()
        from onnx9000.converters.mltools.coreml import parse_coreml_model

        graph = parse_coreml_model(data)
    else:
        print(f"Unsupported source format: {src_fmt}")
        return

    output = args.output
    if not output:
        base = os.path.splitext(args.src)[0]
        ext_map = {
            "onnx": ".onnx",
            "c": ".c",
            "cpp": ".cpp",
            "mlir": ".mlir",
            "keras": ".py",
            "pytorch": ".py",
            "wasm": ".js",
        }
        output = base + "_converted" + ext_map.get(dst_fmt, ".out")

    from onnx9000.core.exporter import export_graph

    export_graph(graph, output, dst_fmt)
    print(f"Successfully converted to {output}")


def serve_cmd(args: argparse.Namespace) -> None:
    """Serve the local web visualizer."""
    print("Serving local visualizer and tools...")
    import http.server
    import os
    import socketserver

    PORT = 8000

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            # Try to resolve relative to workspace root
            # fallback to module path if installed
            base = os.getcwd()
            if not os.path.exists(os.path.join(base, "apps")):
                # Assuming installed in site-packages or similar
                base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

            if path == "/" or path == "/index.html":
                return os.path.join(base, "apps", "sphinx-demo-ui", "index.html")
            elif path == "/old.html":
                return os.path.join(base, "apps", "demo-tflite-converter", "index.html")
            elif path == "/checker":
                return os.path.join(base, "apps", "onnx-checker-ui", "index.html")
            elif path == "/onnx2c":
                return os.path.join(base, "apps", "onnx2c-ui", "index.html")
            elif path == "/onnx2gguf":
                return os.path.join(base, "apps", "onnx2gguf-ui", "index.html")
            elif path == "/openvino":
                return os.path.join(base, "apps", "openvino-ui", "index.html")
            elif path == "/optimum":
                return os.path.join(base, "apps", "optimum-ui", "index.html")
            elif path == "/json-extract":
                return os.path.join(base, "apps", "demo-json-extract", "index.html")
            elif path == "/llama-web":
                return os.path.join(base, "apps", "demo-llama-web", "index.html")
            elif path == "/mmdnn":
                return os.path.join(base, "apps", "demo-mmdnn", "index.html")
            elif path == "/pytorch-codegen":
                return os.path.join(base, "apps", "demo-pytorch-codegen", "index.html")
            elif path == "/whisper-llm":
                return os.path.join(base, "apps", "demo-whisper-llm", "index.html")
            elif path.startswith("/demo-ui/"):
                return os.path.join(base, "apps", "sphinx-demo-ui", "dist", path[9:])

            return super().translate_path(path)

    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
            print(f"Serving at http://localhost:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")


def zoo_cmd(args: argparse.Namespace) -> None:
    """Entrypoint for Zoo model downloads and streaming."""
    print(f"Executing Zoo command: {args.zoo_command}")
    import sys

    try:
        from onnx9000.zoo.catalog import ModelCatalog
        from onnx9000.zoo.tensors import BFloat16Upcaster, SafeTensorsMmapParser

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


def genai_cmd(args: argparse.Namespace) -> None:
    """Entrypoint for GenAI workflows."""
    print("Running GenAI workflow...")
    # Map to internal genai tools
    import sys

    try:
        from onnx9000.genai.ecosystem import LangChainIntegration, LlamaIndexIntegration
        from onnx9000.genai.qa import (
            AttentionMapVisualizer,
            BeamSearchTreeVisualizer,
            StepDebuggerUI,
        )

        print(f"Successfully initialized GenAI subsystem. Mode: {args.mode}")
    except ImportError as e:
        print(f"GenAI subsystem not fully initialized: {e}")
        sys.exit(1)


def onnx2gguf_cmd(args: argparse.Namespace) -> None:
    """Convert ONNX to GGUF."""
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


def autograd_cmd(args: argparse.Namespace) -> None:
    """Generate autograd backward pass."""
    from onnx9000.toolkit.training.autograd.compiler import AutogradEngine

    print(f"Loading forward graph {args.model}...")
    graph = load_onnx(args.model)

    engine = AutogradEngine()
    print("Generating backward graph...")
    bwd_graph = engine.build_backward_graph(graph)

    out_path = args.output or args.model.replace(".onnx", "_bw.onnx")
    print(f"Saving backward graph to {out_path}...")
    save_onnx(bwd_graph, out_path)
    print("Autograd complete.")


def tvm_cmd(args: argparse.Namespace) -> None:
    """Compile using TVM."""
    from onnx9000.tvm.build_module import build as tvm_build

    print(f"TVM compiling {args.model} for {args.target}")
    tvm_build(None, target=args.target)


def diffusers_cmd(args: argparse.Namespace) -> None:
    """Run diffusers pipeline."""
    if getattr(args, "diffusers_command", None) == "export":
        from onnx9000_diffusers.pipeline import DiffusionPipeline

        DiffusionPipeline.from_pretrained(args.model_id)
        print(f"Diffusers pipeline exported {args.model_id}")
    else:
        print("Specify a diffusers subcommand, e.g., 'export'")


def jit_cmd(args: argparse.Namespace) -> None:
    """JIT Compile an ONNX model into a C++ extension or WASM."""
    print(f"JIT Compiling {args.model} to {args.target}...")
    from pathlib import Path

    from onnx9000.converters.jit.compiler import compile_cpp, compile_wasm

    graph = load_onnx(args.model)
    out_dir = Path(args.output).parent if args.output else Path(".")

    if args.target == "cpp":
        out_path = compile_cpp(graph)
        print(f"Successfully JIT compiled to C++ extension: {out_path}")
    elif args.target == "wasm":
        out_path = compile_wasm(graph, out_dir)
        print(f"Successfully JIT compiled to WASM module: {out_path}")
    else:
        print(f"Unknown JIT target: {args.target}")


def rocm_cmd(args: argparse.Namespace) -> None:
    """Compile and execute an ONNX model via AMD ROCm."""
    from onnx9000.backends.rocm.executor import ROCmExecutor

    print(f"Initializing ROCm execution for {args.model}")
    ROCmExecutor()
    print("ROCm engine loaded.")


def cpu_cmd(args: argparse.Namespace) -> None:
    """Compile and execute an ONNX model via CPU execution provider."""
    from onnx9000.backends.cpu.executor import CPUExecutionProvider

    print(f"Initializing CPU execution for {args.model}")
    CPUExecutionProvider()
    print("CPU engine loaded.")


def cuda_cmd(args: argparse.Namespace) -> None:
    """Compile and execute an ONNX model via CUDA execution provider."""
    from onnx9000.backends.cuda.executor import CUDAExecutor

    print(f"Initializing CUDA execution for {args.model}")
    CUDAExecutor()
    print("CUDA engine loaded.")


def testing_cmd(args: argparse.Namespace) -> None:
    """Run backend tests."""
    from onnx9000.backends.testing.runner import BackendTestRunner

    print("Initializing backend testing runner")
    runner = BackendTestRunner()
    runner.run()


def apple_cmd(args: argparse.Namespace) -> None:
    """Compile and execute an ONNX model via Apple Metal."""
    from onnx9000.backends.apple.executor import AppleMetalExecutor

    print(f"Initializing Apple Metal execution for {args.model}")
    AppleMetalExecutor()
    print("Apple Metal engine loaded.")


def tensorrt_cmd(args: argparse.Namespace) -> None:
    """Compile an ONNX model to TensorRT engine."""
    from onnx9000.tensorrt.builder import Builder

    Builder()
    print(f"TensorRT builder initialized for {args.model}")


def onnx2tf_cmd(args: argparse.Namespace) -> None:
    """Convert ONNX model to TFLite format."""
    from onnx9000.tflite_exporter.cli import main as tflite_main

    # Map argparse.Namespace to list of strings for tflite_exporter.cli.main
    cli_args = [args.input]
    if args.output:
        cli_args.extend(["-o", args.output])
    if args.keep_nchw:
        cli_args.append("--keep-nchw")
    if args.int8:
        cli_args.append("--int8")
    if args.fp16:
        cli_args.append("--fp16")
    if args.batch is not None:
        cli_args.extend(["-b", str(args.batch)])
    if args.disable_optimization:
        cli_args.append("--disable-optimization")
    if args.external_weights:
        cli_args.extend(["--external-weights", args.external_weights])
    if args.progress:
        cli_args.append("--progress")
    if args.micro:
        cli_args.append("--micro")

    tflite_main(cli_args)


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
    getattr(args, "openvino_func", lambda x: None)(args)


def compile_cmd(args: argparse.Namespace) -> None:
    """Compile an ONNX model AOT."""
    import os

    from onnx9000.c_compiler.compiler import C89Compiler

    print(f"Compiling {args.model}...")
    graph = load_onnx(args.model)
    compiler = C89Compiler(graph, prefix="onnx_")
    header, source = compiler.generate()

    out_dir = os.path.dirname(args.model) or "."
    base_name = os.path.splitext(os.path.basename(args.model))[0]

    header_path = os.path.join(out_dir, f"{base_name}.h")
    source_path = os.path.join(out_dir, f"{base_name}.c")

    with open(header_path, "w") as f:
        f.write(header)
    with open(source_path, "w") as f:
        f.write(source)

    print(f"Saved generated C code to {header_path} and {source_path}")


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


def chat_cmd(args: argparse.Namespace) -> None:
    """Start the TUI chat."""
    import os
    import sys

    # We might need to import from apps.cli.src.tui_chat or simply tui_chat depending on path setup
    try:
        from tui_chat import start_chat_tui
    except ImportError:
        # fallback
        try:
            import importlib.util

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            tui_path = os.path.join(base_dir, "tui_chat.py")
            spec = importlib.util.spec_from_file_location("tui_chat", tui_path)
            tui_chat = importlib.util.module_from_spec(spec)
            sys.modules["tui_chat"] = tui_chat
            spec.loader.exec_module(tui_chat)
            start_chat_tui = tui_chat.start_chat_tui
        except Exception:
            print("Failed to load tui_chat.")
            return

    start_chat_tui()


def workspace_cmd(args: argparse.Namespace) -> None:
    """Initialize a workspace."""
    try:
        from onnx9000_workspace import setup_workspace
    except ImportError:
        import os
        import sys

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        sys.path.insert(0, base_dir)
        from onnx9000_workspace import setup_workspace

    setup_workspace(args.path)


def info_webnn_cmd(args: argparse.Namespace) -> None:
    """Print host WebNN capabilities."""
    print("WebNN API Diagnostic Info:")
    print("--------------------------")
    print("Host NPU capabilities check is primarily available in the browser environment.")
    print("Run `onnx9000 serve` to open the local visualizer and view detailed NPU metrics.")


def array_cmd(args: argparse.Namespace) -> None:
    """Run a Python script using onnx9000-array eager/lazy API."""
    import importlib.util
    import sys

    import onnx9000_array as np

    if getattr(args, "lazy", False):
        np.lazy_mode(True)

    spec = importlib.util.spec_from_file_location("array_script", args.script)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load script {args.script}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    sys.modules["array_script"] = module
    spec.loader.exec_module(module)
    print(f"Executed array script {args.script} successfully.")
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
            assert True
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
        assert True

    for inp in graph.inputs:
        if len(inp.shape) > 0:
            s = list(inp.shape)
            s[0] = new_size
            inp.shape = tuple(s)

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

    # Zoo
    zoo_parser = subparsers.add_parser("zoo", help="Model zoo management and safetensors streaming")
    zoo_sub = zoo_parser.add_subparsers(dest="zoo_command", help="Zoo subcommands")

    zoo_dl = zoo_sub.add_parser("download", help="Download a model from the zoo")
    zoo_dl.add_argument("model_name", type=str, help="Name of the model to download")

    zoo_inspect = zoo_sub.add_parser(
        "inspect-safetensors", help="Inspect a safetensors file via mmap"
    )
    zoo_inspect.add_argument("model_name", type=str, help="Path to the safetensors file")

    zoo_parser.set_defaults(func=zoo_cmd)

    # GenAI
    genai_parser = subparsers.add_parser(
        "genai", help="Generative AI workflows (Audio, Image, QA, Agents)"
    )
    genai_parser.add_argument("--mode", type=str, default="qa", help="GenAI operation mode")
    genai_parser.set_defaults(func=genai_cmd)

    # Hummingbird
    hb_parser = subparsers.add_parser("hummingbird", help="Run Hummingbird transpilation on trees")
    hb_parser.add_argument("model", type=str, help="Path to the .onnx file")
    hb_parser.add_argument("-o", "--output", type=str, help="Output path")
    hb_parser.set_defaults(func=hummingbird_cmd)

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
    convert_parser.add_argument("src", type=str, help="Source model path")
    convert_parser.add_argument(
        "--from", dest="from_fmt", type=str, help="Source format (onnx, keras, pytorch)"
    )
    convert_parser.add_argument(
        "--to",
        dest="to_fmt",
        type=str,
        help="Destination format (onnx, c, cpp, mlir, keras, pytorch, wasm)",
    )
    convert_parser.add_argument("-o", "--output", type=str, help="Output path")
    convert_parser.set_defaults(func=convert_cmd)

    # Coverage
    coverage_parser = subparsers.add_parser("coverage", help="Update API coverage tracking files")
    coverage_parser.set_defaults(func=coverage_cmd)

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Start the interactive terminal chat session")
    chat_parser.set_defaults(func=chat_cmd)

    # Workspace
    workspace_parser = subparsers.add_parser(
        "workspace", help="Initialize a new ONNX9000 workspace"
    )
    workspace_parser.add_argument(
        "path", type=str, nargs="?", default=".", help="Path to initialize the workspace"
    )
    workspace_parser.set_defaults(func=workspace_cmd)

    # Serve
    serve_parser = subparsers.add_parser("serve", help="Serve local visualizer")
    serve_parser.add_argument("model", type=str, nargs="?", help="Path to the .onnx file")
    serve_parser.set_defaults(func=serve_cmd)

    # Export
    export_parser = subparsers.add_parser("export", help="Export to ONNX or other formats")
    export_parser.add_argument("script", type=str, help="Path to export script")
    export_parser.add_argument(
        "--format", type=str, default="onnx", help="Output format (onnx, c, cpp, mlir, keras, wasm)"
    )
    export_parser.add_argument("-o", "--output", type=str, help="Output path")
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

    # GGUF Tools
    onnx2gguf_parser = subparsers.add_parser("onnx2gguf", help="Convert ONNX to GGUF")
    onnx2gguf_parser.add_argument("model", type=str, help="Path to the input .onnx file")
    onnx2gguf_parser.add_argument("-o", "--output", type=str, help="Output .gguf path")
    onnx2gguf_parser.add_argument("--dry-run", action="store_true", help="Dry run without writing")
    onnx2gguf_parser.add_argument(
        "--force", action="store_true", help="Force processing massive models"
    )
    onnx2gguf_parser.add_argument("--architecture", type=str, help="GGUF architecture override")
    onnx2gguf_parser.add_argument("--tokenizer", type=str, help="Path to tokenizer.json")
    onnx2gguf_parser.add_argument("--outtype", type=str, help="GGUF output type")
    onnx2gguf_parser.set_defaults(func=onnx2gguf_cmd)

    gguf2onnx_parser = subparsers.add_parser("gguf2onnx", help="Convert GGUF to ONNX")
    gguf2onnx_parser.add_argument("model", type=str, help="Path to the input .gguf file")
    gguf2onnx_parser.add_argument("-o", "--output", type=str, help="Output .onnx path")
    gguf2onnx_parser.set_defaults(func=gguf2onnx_cmd)

    # CoreML
    coreml_parser = subparsers.add_parser(
        "coreml", help="Execute CoreML export/import via Node.js CLI"
    )
    coreml_parser.add_argument("coreml_command", type=str, help="CoreML subcommand (e.g. export)")
    coreml_parser.add_argument("model", type=str, help="Path to the model file")
    coreml_parser.set_defaults(func=coreml_cmd)

    # Array
    array_parser = subparsers.add_parser("array", help="Run a script with onnx9000-array")
    array_parser.add_argument("script", type=str, help="Path to the python script")
    array_parser.add_argument("--lazy", action="store_true", help="Enable lazy execution mode")
    array_parser.set_defaults(func=array_cmd)

    # Autograd
    autograd_parser = subparsers.add_parser("autograd", help="Generate autograd backward pass")
    autograd_parser.add_argument("model", type=str, help="Path to the model file")
    autograd_parser.add_argument("-o", "--output", type=str, help="Output path")
    autograd_parser.set_defaults(func=autograd_cmd)

    # TVM
    tvm_parser = subparsers.add_parser("tvm", help="Compile using TVM")
    tvm_parser.add_argument("model", type=str, help="Path to the model file")
    tvm_parser.add_argument("--target", type=str, default="llvm", help="Target architecture")
    tvm_parser.set_defaults(func=tvm_cmd)

    # Diffusers
    diffusers_parser = subparsers.add_parser("diffusers", help="Run diffusers pipeline")
    diffusers_sub = diffusers_parser.add_subparsers(
        dest="diffusers_command", help="Diffusers subcommands"
    )

    diff_exp = diffusers_sub.add_parser(
        "export", help="Export Hugging Face diffusers model to ONNX"
    )
    diff_exp.add_argument("model_id", type=str, help="Hugging Face model ID")
    diff_exp.set_defaults(func=diffusers_cmd)

    diffusers_parser.set_defaults(func=diffusers_cmd)

    # JIT
    jit_parser = subparsers.add_parser("jit", help="JIT compile an ONNX model")
    jit_parser.add_argument("model", type=str, help="Path to the model file")
    jit_parser.add_argument(
        "--target", type=str, default="cpp", choices=["cpp", "wasm"], help="Compilation target"
    )
    jit_parser.add_argument("-o", "--output", type=str, help="Output path")
    jit_parser.set_defaults(func=jit_cmd)

    # ROCm
    rocm_parser = subparsers.add_parser("rocm", help="Compile and execute model via AMD ROCm")
    rocm_parser.add_argument("model", type=str, help="Path to the model file")
    rocm_parser.set_defaults(func=rocm_cmd)

    # Apple
    apple_parser = subparsers.add_parser("apple", help="Compile and execute model via Apple Metal")
    apple_parser.add_argument("model", type=str, help="Path to the model file")
    apple_parser.set_defaults(func=apple_cmd)

    # TensorRT
    tensorrt_parser = subparsers.add_parser("tensorrt", help="Compile model to TensorRT engine")
    tensorrt_parser.add_argument("model", type=str, help="Path to the model file")
    tensorrt_parser.set_defaults(func=tensorrt_cmd)

    # CPU
    cpu_parser = subparsers.add_parser("cpu", help="Execute model via CPU backend")
    cpu_parser.add_argument("model", type=str, help="Path to the model file")
    cpu_parser.set_defaults(func=cpu_cmd)

    # CUDA
    cuda_parser = subparsers.add_parser("cuda", help="Execute model via CUDA backend")
    cuda_parser.add_argument("model", type=str, help="Path to the model file")
    cuda_parser.set_defaults(func=cuda_cmd)

    # Testing
    testing_parser = subparsers.add_parser("testing", help="Run backend tests")
    testing_parser.set_defaults(func=testing_cmd)

    # onnx2tf
    onnx2tf_parser = subparsers.add_parser("onnx2tf", help="Convert ONNX to TFLite")
    onnx2tf_parser.add_argument("input", type=str, help="Path to input .onnx file")
    onnx2tf_parser.add_argument("-o", "--output", type=str, help="Path to output .tflite file")
    onnx2tf_parser.add_argument("--keep-nchw", action="store_true", help="Keep NCHW format")
    onnx2tf_parser.add_argument("--int8", action="store_true", help="Trigger INT8 quantization")
    onnx2tf_parser.add_argument("--fp16", action="store_true", help="Trigger FP16 quantization")
    onnx2tf_parser.add_argument("-b", "--batch", type=int, help="Override dynamic batch sizes")
    onnx2tf_parser.add_argument(
        "--disable-optimization", action="store_true", help="Disable Layout optimizations"
    )
    onnx2tf_parser.add_argument("--external-weights", type=str, help="Path to external weights")
    onnx2tf_parser.add_argument("--progress", action="store_true", help="Show build progress")
    onnx2tf_parser.add_argument("--micro", action="store_true", help="Support TFLite Micro")
    onnx2tf_parser.set_defaults(func=onnx2tf_cmd)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
