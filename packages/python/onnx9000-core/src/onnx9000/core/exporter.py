"""Unified Exporter for ONNX9000 IR Graphs."""

import os
from onnx9000.core.ir import Graph
from onnx9000.core.serializer import save as save_onnx


def export_graph(graph: Graph, output_path: str, format: str):
    """Export an IR Graph to the specified format and save to output_path."""
    from onnx9000.core.shape_inference import infer_shapes_and_types

    infer_shapes_and_types(graph)

    format = format.lower()
    if format == "onnx":
        save_onnx(graph, output_path)
    elif format == "c":
        from onnx9000.c_compiler.compiler import C89Compiler

        compiler = C89Compiler(graph)
        compiler._generate_header()
        compiler._generate_source()

        with open(output_path, "w") as f:
            f.write(compiler.source_builder.get_code())

        header_path = output_path.replace(".c", ".h")
        if header_path == output_path:
            header_path += ".h"
        with open(header_path, "w") as f:
            f.write(compiler.header_builder.get_code())

    elif format == "cpp":
        from onnx9000.backends.codegen.generator import Generator

        generator = Generator(graph)
        code = generator.generate()
        with open(output_path, "w") as f:
            f.write(code)

    elif format == "wasm":
        from onnx9000.converters.jit.compiler import compile_wasm
        from pathlib import Path

        # compile_wasm saves to a file and returns the path
        js_path = compile_wasm(graph, Path(os.path.dirname(output_path) or "."))
        if str(js_path) != output_path:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(js_path, output_path)
            wasm_path = str(js_path).replace(".js", ".wasm")
            if os.path.exists(wasm_path):
                target_wasm = output_path.replace(".js", ".wasm")
                if os.path.exists(target_wasm):
                    os.remove(target_wasm)
                os.rename(wasm_path, target_wasm)

    elif format == "keras":
        code = generate_keras(graph)
        with open(output_path, "w") as f:
            f.write(code)

    elif format == "mlir":
        code = generate_mlir(graph)
        with open(output_path, "w") as f:
            f.write(code)
    else:
        raise ValueError(f"Unsupported format: {format}")


def generate_keras(graph: Graph) -> str:
    """Generate Keras 3 (Python) code from an IR Graph."""
    lines = [
        "import keras",
        "import numpy as np",
        "from keras import ops",
        "from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Add, Activation, BatchNormalization, Lambda, Permute, SimpleRNN, LSTM, GRU, Dot",
        "",
    ]

    class_name = f"Model_{graph.name or 'Generated'}"
    lines.append(f"class {class_name}(keras.Model):")
    lines.append("    def __init__(self, **kwargs):")
    lines.append("        super().__init__(**kwargs)")

    init_lines = []
    call_lines = []

    # Very simple mapping for a few common ops
    layer_counter = 0

    # Inputs handling
    inputs_str = ", ".join([f"in_{i}" for i in range(len(graph.inputs))])
    if len(graph.inputs) > 1:
        call_lines.append(f"        ({inputs_str}) = inputs")
    elif len(graph.inputs) == 1:
        call_lines.append(f"        in_0 = inputs")
    else:
        call_lines.append("        pass")

    tensor_map = {name: f"in_{i}" for i, name in enumerate(graph.inputs)}

    for node in graph.nodes:
        out_name = f"node_{layer_counter}"
        layer_counter += 1

        inps = [tensor_map.get(i, "None") for i in node.inputs]

        if node.op_type == "Conv":
            l_name = f"conv_{layer_counter}"
            init_lines.append(
                f"        self.{l_name} = Conv2D(filters=32, kernel_size=(3,3), padding='same')"
            )
            call_lines.append(f"        {out_name} = self.{l_name}({inps[0]})")
        elif node.op_type == "Relu":
            call_lines.append(f"        {out_name} = ops.relu({inps[0]})")
        elif node.op_type == "Add":
            call_lines.append(f"        {out_name} = {inps[0]} + {inps[1]}")
        elif node.op_type == "MatMul" or node.op_type == "Gemm":
            l_name = f"dense_{layer_counter}"
            init_lines.append(f"        self.{l_name} = Dense(units=10)")
            call_lines.append(f"        {out_name} = self.{l_name}({inps[0]})")
        else:
            call_lines.append(
                f"        {out_name} = {inps[0]}  # Identity fallback for {node.op_type}"
            )

        for o in node.outputs:
            tensor_map[o] = out_name

    if not init_lines:
        init_lines.append("        pass")

    for l in init_lines:
        lines.append(l)

    lines.append("")
    lines.append("    def call(self, inputs):")
    for l in call_lines:
        lines.append(l)

    outputs_str = ", ".join([tensor_map.get(o, "None") for o in graph.outputs])
    lines.append(f"        return [{outputs_str}]")

    return "\n".join(lines) + "\n"


def generate_mlir(graph: Graph) -> str:
    """Generate a basic MLIR representation of the Graph."""
    lines = ["module {", "  func.func @main("]

    # Inputs
    inputs_text = []
    for i, name in enumerate(graph.inputs):
        t = graph.tensors.get(name)
        shape = "x".join(map(str, t.shape)) if t and t.shape else "tensor<*xf32>"
        inputs_text.append(f"%arg{i}: tensor<{shape}xf32>")

    lines[1] += ", ".join(inputs_text) + ") -> ("

    # Outputs types
    outputs_types = []
    for name in graph.outputs:
        t = graph.tensors.get(name)
        shape = "x".join(map(str, t.shape)) if t and t.shape else "tensor<*xf32>"
        outputs_types.append(f"tensor<{shape}xf32>")

    lines[1] += ", ".join(outputs_types) + ") {"

    # Body
    val_counter = len(graph.inputs)
    tensor_map = {name: f"%arg{i}" for i, name in enumerate(graph.inputs)}

    for node in graph.nodes:
        inps = [tensor_map.get(inp, f"%cst_{val_counter}") for inp in node.inputs]
        outs = [f"%v{val_counter + i}" for i in range(len(node.outputs))]
        val_counter += len(node.outputs)

        lines.append(
            f'    {", ".join(outs)} = "onnx.{node.op_type}"({", ".join(inps)}) : ({", ".join(["tensor<*xf32>"] * len(inps))}) -> ({", ".join(["tensor<*xf32>"] * len(outs))})'
        )

        for i, o in enumerate(node.outputs):
            tensor_map[o] = outs[i]

    # Return
    ret_vals = [tensor_map.get(o, "None") for o in graph.outputs]
    lines.append(f"    return {', '.join(ret_vals)} : {', '.join(outputs_types)}")

    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"
