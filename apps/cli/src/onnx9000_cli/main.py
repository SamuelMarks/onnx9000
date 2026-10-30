"""ONNX9000 Unified MLOps and Execution Ecosystem CLI Entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
import time

from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.core.parser.core import load as load_onnx
from onnx9000.core.serializer import save as save_onnx
from onnx9000.optimizer.simplifier import simplify

from onnx9000_cli.commands.compile import (
    apple_cmd,
    array_cmd,
    autograd_cmd,
    compile_cmd,
    cpu_cmd,
    cuda_cmd,
    custom_ops_cmd,
    iree_cmd,
    jit_cmd,
    mlir_cmd,
    ort_training_cmd,
    rocm_cmd,
    tensorrt_cmd,
    testing_cmd,
    triton_cmd,
    tvm_cmd,
    wasm_cmd,
    webgpu_cmd,
    zero_dep_classifier_cmd,
)
from onnx9000_cli.commands.convert import (
    convert_cmd,
    coreml_cmd,
    diffusers_cmd,
    export_cmd,
    gguf2onnx_cmd,
    jax_cmd,
    keras2onnx_cmd,
    mmdnn_cmd,
    onnx2c_cmd,
    onnx2gguf_cmd,
    onnx2tf_cmd,
    openvino_cmd,
    openvino_export_cmd,
    paddle2onnx_cmd,
    pytorch_codegen_cmd,
    script_cmd,
    skl2onnx_cmd,
)
from onnx9000_cli.commands.genai import (
    agent_cmd,
    chat_cmd,
    genai_cmd,
    llama_web_cmd,
    transformers_cmd,
    whisper_llm_cmd,
)
from onnx9000_cli.commands.optimize import (
    hummingbird_cmd,
    olive_optimizer_cmd,
    optimize_cmd,
    optimum_cmd,
    optimum_export_cmd,
    optimum_optimize_cmd,
    optimum_quantize_cmd,
    prune_cmd,
    quantize_cmd,
    sparse_cmd,
    sparse_de_sparsify_cmd,
    sparse_prune_cmd,
)
from onnx9000_cli.commands.profile import (
    arena_cmd,
    coverage_cmd,
    info_cmd,
    info_webnn_cmd,
    mobile_memory_cmd,
    onnx_tool_cmd,
    profiler_cmd,
)
from onnx9000_cli.commands.serve import (
    serve_cmd,
    sphinx_demo_ui_cmd,
    tfjs_shim_cmd,
    triton_server_cmd,
    webnn_polyfill_cmd,
)
from onnx9000_cli.commands.simplify import (
    change_batch_cmd,
    edit_cmd,
    graphsurgeon_cmd,
    inspect_cmd,
    json_extract_cmd,
    mutate_cmd,
    onnx_checker_cmd,
    rename_input_cmd,
    simplify_cmd,
)
from onnx9000_cli.commands.zoo import (
    new_model_arch_cmd,
    progressive_loading_cmd,
    workspace_cmd,
    zoo_cmd,
)


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(
        prog="onnx9000",
        description="ONNX9000 Unified MLOps and Execution Ecosystem CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Custom Ops
    custom_ops_parser = subparsers.add_parser("custom-ops", help="Register custom ops")
    custom_ops_parser.add_argument("ops_file", type=str, help="Path to the custom ops file")
    custom_ops_parser.set_defaults(func=custom_ops_cmd)

    # Profiler
    profiler_parser = subparsers.add_parser("profiler", help="Run Memory Arena & Profiler")
    profiler_parser.add_argument("model", type=str, help="Path to the ONNX model")
    profiler_parser.add_argument(
        "--show-arena", action="store_true", help="Show memory arena visualization"
    )
    profiler_parser.set_defaults(func=profiler_cmd)

    ort_training_parser = subparsers.add_parser("ort-training", help="ORT Training")
    ort_training_parser.add_argument("model", type=str)
    ort_training_parser.set_defaults(func=ort_training_cmd)

    olive_optimizer_parser = subparsers.add_parser("olive-optimizer", help="Olive Optimizer")
    olive_optimizer_parser.add_argument("model", type=str)
    olive_optimizer_parser.set_defaults(func=olive_optimizer_cmd)

    triton_server_parser = subparsers.add_parser("triton-server", help="Triton Server")
    triton_server_parser.add_argument("model", type=str)
    triton_server_parser.set_defaults(func=triton_server_cmd)

    onnx_tool_parser = subparsers.add_parser("onnx-tool", help="ONNX Tool")
    onnx_tool_parser.add_argument("model", type=str)
    onnx_tool_parser.set_defaults(func=onnx_tool_cmd)

    # Transformers
    transformers_parser = subparsers.add_parser("transformers", help="Run Transformers.js Pipeline")
    transformers_parser.add_argument(
        "task", type=str, help="The pipeline task (e.g. text-classification)"
    )
    transformers_parser.add_argument(
        "inputs", nargs=argparse.REMAINDER, help="Input string for the pipeline"
    )
    transformers_parser.set_defaults(func=transformers_cmd)

    # Inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect an ONNX model")
    inspect_parser.add_argument("model", type=str, help="Path to the .onnx file")
    inspect_parser.set_defaults(func=inspect_cmd)

    # PyTorch Codegen
    pytorch_codegen_parser = subparsers.add_parser(
        "pytorch-codegen", help="Generate PyTorch code from an ONNX model"
    )
    pytorch_codegen_parser.add_argument("model", type=str, help="Path to the .onnx file")
    pytorch_codegen_parser.add_argument("-o", "--output", type=str, help="Path to output .py file")
    pytorch_codegen_parser.set_defaults(func=pytorch_codegen_cmd)

    # Whisper LLM
    whisper_llm_parser = subparsers.add_parser("whisper-llm", help="Run Whisper transcription")
    whisper_llm_parser.add_argument("model", type=str, help="Path to the Whisper ONNX model")
    whisper_llm_parser.add_argument("audio", type=str, help="Path to the audio file")
    whisper_llm_parser.add_argument("-o", "--output", type=str, help="Path to save transcription")
    whisper_llm_parser.set_defaults(func=whisper_llm_cmd)

    # LLaMA Web
    llama_web_parser = subparsers.add_parser("llama-web", help="Run LLaMA generation")
    llama_web_parser.add_argument("model", type=str, help="Path to the LLaMA ONNX model")
    llama_web_parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    llama_web_parser.add_argument("-o", "--output", type=str, help="Path to save output text")
    llama_web_parser.set_defaults(func=llama_web_cmd)

    # TFJS Shim

    webnn_polyfill_parser = subparsers.add_parser(
        "webnn-polyfill", help="Run WebNN Polyfill diagnostic"
    )
    webnn_polyfill_parser.set_defaults(func=webnn_polyfill_cmd)

    tfjs_shim_parser = subparsers.add_parser("tfjs-shim", help="Run TFJS Shim diagnostic")
    tfjs_shim_parser.set_defaults(func=tfjs_shim_cmd)

    # Triton
    triton_parser = subparsers.add_parser("triton", help="Triton Compiler")
    triton_parser.add_argument("model", type=str, help="Path to the ONNX model")
    triton_parser.set_defaults(func=triton_cmd)

    # TensorRT
    tensorrt_parser = subparsers.add_parser("tensorrt", help="TensorRT Exporter")
    tensorrt_parser.add_argument("model", type=str, help="Path to the ONNX model")
    tensorrt_parser.set_defaults(func=tensorrt_cmd)

    # IREE
    iree_parser = subparsers.add_parser("iree", help="IREE Compiler & Runtime")
    iree_sub = iree_parser.add_subparsers(dest="iree_command", help="IREE subcommands")

    iree_compile = iree_sub.add_parser("compile", help="Compile model")
    iree_compile.add_argument("model", type=str, help="Path to the ONNX model")

    iree_run = iree_sub.add_parser("run", help="Run model")
    iree_run.add_argument("module", type=str, help="Path to the compiled module")

    iree_parser.set_defaults(func=iree_cmd)

    # JSON Extract
    json_extract_parser = subparsers.add_parser(
        "json-extract", help="Extract JSON representation of an ONNX model"
    )
    json_extract_parser.add_argument("model", type=str, help="Path to the .onnx file")
    json_extract_parser.add_argument("-o", "--output", type=str, help="Path to output .json file")
    json_extract_parser.set_defaults(func=json_extract_cmd)

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
        "--target-opset",
        type=int,
        help="Target ONNX opset version to explicitly override",
    )
    simplify_parser.add_argument(
        "--tensor-type",
        type=str,
        action="append",
        help="Input type override, e.g., 'x:FLOAT32'",
    )
    simplify_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    simplify_parser.add_argument(
        "--strip-metadata", action="store_true", help="Strip model metadata"
    )
    simplify_parser.add_argument(
        "--sort-value-info",
        action="store_true",
        help="Sort ValueInfo lists alphabetically",
    )
    simplify_parser.add_argument(
        "--diff-json",
        action="store_true",
        help="Output a visual DAG difference JSON file",
    )
    simplify_parser.add_argument(
        "--preserve-nodes",
        type=str,
        help="Comma-separated names of nodes to preserve from DCE",
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
        "--from",
        dest="from_fmt",
        type=str,
        help="Source format (onnx, keras, pytorch, darknet, ncnn, caffe, cntk, mxnet, sklearn, xgboost, catboost, lightgbm, pyspark, paddle, jax, flax, tensorflow)",
    )
    convert_parser.add_argument(
        "--to",
        dest="to_fmt",
        type=str,
        help="Destination format (onnx, c, cpp, mlir, keras, pytorch, wasm)",
    )
    convert_parser.add_argument("-o", "--output", type=str, help="Output path")
    convert_parser.add_argument(
        "--weights", type=str, help="Path to weights file (e.g., for Darknet)"
    )
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
        "path",
        type=str,
        nargs="?",
        default=".",
        help="Path to initialize the workspace",
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
        "--format",
        type=str,
        default="onnx",
        help="Output format (onnx, c, cpp, mlir, keras, wasm)",
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
        "--shape",
        type=str,
        action="append",
        help="Shape override, e.g., 'input:[1,3,224,224]'",
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
        "--target",
        type=str,
        default="cpp",
        choices=["cpp", "wasm"],
        help="Compilation target",
    )
    jit_parser.add_argument("-o", "--output", type=str, help="Output path")
    jit_parser.set_defaults(func=jit_cmd)

    # ROCm
    rocm_parser = subparsers.add_parser("rocm", help="Compile and execute model via AMD ROCm")
    rocm_parser.add_argument("model", type=str, help="Path to the model file")
    rocm_parser.set_defaults(func=rocm_cmd)

    # Apple
    # WebGPU

    wasm_parser = subparsers.add_parser("wasm", help="Execute model via WebAssembly backend")
    wasm_parser.add_argument("model", type=str, help="Path to the model file")
    wasm_parser.set_defaults(func=wasm_cmd)

    webgpu_parser = subparsers.add_parser("webgpu", help="Execute model via WebGPU backend")
    webgpu_parser.add_argument("model", type=str, help="Path to the model file")
    webgpu_parser.set_defaults(func=webgpu_cmd)

    apple_parser = subparsers.add_parser("apple", help="Compile and execute model via Apple Metal")
    apple_parser.add_argument("model", type=str, help="Path to the model file")
    apple_parser.set_defaults(func=apple_cmd)

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

    # Sphinx Demo UI
    sphinx_ui_parser = subparsers.add_parser("sphinx-demo-ui", help="Launch Sphinx Demo UI")
    sphinx_ui_parser.set_defaults(func=sphinx_demo_ui_cmd)

    # ONNX Checker
    checker_parser = subparsers.add_parser("onnx-checker", help="Check ONNX model validity")
    checker_parser.add_argument("model", type=str, help="Path to input .onnx file")
    checker_parser.set_defaults(func=onnx_checker_cmd)

    # MLIR Lowering
    mlir_parser = subparsers.add_parser("mlir", help="Lower ONNX model to MLIR")
    mlir_parser.add_argument("model", type=str, help="Path to input .onnx file")
    mlir_parser.set_defaults(func=mlir_cmd)

    # Mobile Memory
    mobile_memory_parser = subparsers.add_parser("mobile-memory", help="Analyze mobile memory")
    mobile_memory_parser.add_argument("model", type=str, help="Path to input .onnx file")
    mobile_memory_parser.set_defaults(func=mobile_memory_cmd)

    # Progressive Loading
    progressive_loading_parser = subparsers.add_parser(
        "progressive-loading", help="Generate progressive chunks"
    )
    progressive_loading_parser.add_argument("model", type=str, help="Path to input .onnx file")
    progressive_loading_parser.set_defaults(func=progressive_loading_cmd)

    # New Model Arch
    new_model_arch_parser = subparsers.add_parser(
        "new-model-arch", help="Scaffold new model architecture"
    )
    new_model_arch_parser.add_argument("arch", type=str, help="Name of the architecture")
    new_model_arch_parser.set_defaults(func=new_model_arch_cmd)

    # Zero Dep Classifier
    zero_dep_classifier_parser = subparsers.add_parser(
        "zero-dep-classifier", help="Generate zero-dep C classifier"
    )
    zero_dep_classifier_parser.add_argument("model", type=str, help="Path to input .onnx file")
    zero_dep_classifier_parser.set_defaults(func=zero_dep_classifier_cmd)

    # MMDNN
    mmdnn_parser = subparsers.add_parser("mmdnn", help="Convert model via MMDNN")
    mmdnn_parser.add_argument("model", type=str, help="Path to input model")
    mmdnn_parser.set_defaults(func=mmdnn_cmd)

    # JAX
    jax_parser = subparsers.add_parser("jax", help="Convert JAX model to ONNX")
    jax_parser.add_argument("model", type=str, help="Path to input JAX model")
    jax_parser.set_defaults(func=jax_cmd)

    # GraphSurgeon
    gs_parser = subparsers.add_parser("graphsurgeon", help="Modify ONNX graph via script")
    gs_parser.add_argument("model", type=str, help="Path to input .onnx file")
    gs_parser.add_argument("--script", type=str, required=True, help="Path to mutation script")
    gs_parser.set_defaults(func=graphsurgeon_cmd)

    # onnx2c
    onnx2c_parser = subparsers.add_parser("onnx2c", help="Convert ONNX to C source code")
    onnx2c_parser.add_argument("input", type=str, help="Path to input .onnx file")
    onnx2c_parser.add_argument("-o", "--output", type=str, help="Path to output .c file")
    onnx2c_parser.set_defaults(func=onnx2c_cmd)

    # onnx2tf
    onnx2tf_parser = subparsers.add_parser("onnx2tf", help="Convert ONNX to TFLite")
    onnx2tf_parser.add_argument("input", type=str, help="Path to input .onnx file")
    onnx2tf_parser.add_argument("-o", "--output", type=str, help="Path to output .tflite file")
    onnx2tf_parser.add_argument("--keep-nchw", action="store_true", help="Keep NCHW format")
    onnx2tf_parser.add_argument("--int8", action="store_true", help="Trigger INT8 quantization")
    onnx2tf_parser.add_argument("--fp16", action="store_true", help="Trigger FP16 quantization")
    onnx2tf_parser.add_argument("-b", "--batch", type=int, help="Override dynamic batch sizes")
    onnx2tf_parser.add_argument(
        "--disable-optimization",
        action="store_true",
        help="Disable Layout optimizations",
    )
    onnx2tf_parser.add_argument("--external-weights", type=str, help="Path to external weights")
    onnx2tf_parser.add_argument("--progress", action="store_true", help="Show build progress")
    onnx2tf_parser.add_argument("--micro", action="store_true", help="Support TFLite Micro")
    onnx2tf_parser.set_defaults(func=onnx2tf_cmd)

    tflite_parser = subparsers.add_parser("tflite", help="Convert ONNX to TFLite (Alias)")
    tflite_parser.add_argument("input", type=str, help="Path to input .onnx file")
    tflite_parser.add_argument("-o", "--output", type=str, help="Path to output .tflite file")
    tflite_parser.add_argument("--keep-nchw", action="store_true", help="Keep NCHW format")
    tflite_parser.add_argument("--int8", action="store_true", help="Trigger INT8 quantization")
    tflite_parser.add_argument("--fp16", action="store_true", help="Trigger FP16 quantization")
    tflite_parser.add_argument("-b", "--batch", type=int, help="Override dynamic batch sizes")
    tflite_parser.add_argument(
        "--disable-optimization",
        action="store_true",
        help="Disable Layout optimizations",
    )
    tflite_parser.add_argument("--external-weights", type=str, help="Path to external weights")
    tflite_parser.add_argument("--progress", action="store_true", help="Show build progress")
    tflite_parser.add_argument("--micro", action="store_true", help="Support TFLite Micro")
    tflite_parser.set_defaults(func=onnx2tf_cmd)

    agent_parser = subparsers.add_parser("agent", help="Run an autonomous agentic workflow")
    agent_parser.add_argument("task", nargs="*", help="The task for the agent to complete")
    agent_parser.set_defaults(func=agent_cmd)

    script_parser = subparsers.add_parser(
        "script", help="Compile an ONNX Script into an ONNX graph"
    )
    script_parser.add_argument("input", type=str, help="Path to input .py script")
    script_parser.add_argument("-o", "--output", type=str, help="Path to output .onnx file")
    script_parser.set_defaults(func=script_cmd)

    paddle2onnx_parser = subparsers.add_parser("paddle2onnx", help="Paddle2ONNX Converter")
    paddle2onnx_parser.add_argument("model", type=str)
    paddle2onnx_parser.set_defaults(func=paddle2onnx_cmd)

    keras2onnx_parser = subparsers.add_parser("keras2onnx", help="Keras2ONNX Converter")
    keras2onnx_parser.add_argument("model", type=str)
    keras2onnx_parser.set_defaults(func=keras2onnx_cmd)

    skl2onnx_parser = subparsers.add_parser("skl2onnx", help="SKL2ONNX Converter")
    skl2onnx_parser.add_argument("model", type=str)
    skl2onnx_parser.set_defaults(func=skl2onnx_cmd)

    arena_parser = subparsers.add_parser("arena", help="Memory Arena Planner")
    arena_parser.add_argument("model", type=str)
    arena_parser.set_defaults(func=arena_cmd)
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


__all__ = [
    "Attribute",
    "Graph",
    "Node",
    "Tensor",
    "ValueInfo",
    "load_onnx",
    "main",
    "os",
    "save_onnx",
    "simplify",
    "time",
]

if __name__ == "__main__":
    main()
