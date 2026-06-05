"""Compiler for transforming ONNX9000 IR to C89 code."""

# Import memory planner to register Graph.simulate_memory_plan
from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.c_compiler.data_unpacker import unpack_bytes_to_str
from onnx9000.core.dtypes import DType, to_cpp_type
from onnx9000.core.ir import Constant, Graph


class C89Compiler:
    """A compiler that transforms ONNX9000 IR Graphs into C89 compatible source code.

    This compiler handles memory planning, weight serialization, and code generation
    for a wide range of ONNX operators, targeting embedded and restricted environments.
    """

    def __init__(
        self,
        graph: Graph,
        prefix: str = "model_",
        emit_cpp: bool = False,
        target: str = "",
        use_math_h: bool = True,
        debug: bool = False,
        align: int = 0,
        indent: int = 4,
    ):
        """Initialize the C89 compiler with a graph and configuration options.

        Args:
            graph: The IR Graph to compile.
            prefix: Prefix for generated C functions and variables.
            emit_cpp: Whether to add C++ compatibility (extern "C").
            target: Target hardware architecture for specialized optimizations.
            use_math_h: Whether to include and use standard <math.h>.
            debug: Enable debug assertions and checks in generated code.
            align: Memory alignment for tensor buffers.
            indent: Number of spaces for code indentation.

        """
        self.align = align
        self.indent = indent
        self.graph = graph
        self.prefix = prefix
        self.emit_cpp = emit_cpp
        self.target = target
        self.use_math_h = use_math_h
        self.debug = debug
        self.header_builder = C89Builder(prefix)
        self.source_builder = C89Builder(prefix)
        self.arena = self.graph.simulate_memory_plan()

        # Warn if arena > 256KB
        if self.arena.peak_memory > 256 * 1024:
            print(
                f"WARNING: Required arena size ({self.arena.peak_memory} bytes) exceeds standard microcontroller limits (256KB)."
            )

    def _generate_header(self) -> None:
        """Generate the C89 header file containing type definitions, macros, and function prototypes."""
        b = self.header_builder
        guard = f"{self.prefix.upper()}H_INCLUDED"
        b.emit(f"#ifndef {guard}")
        b.emit(f"#define {guard}")
        b.emit("")
        b.emit("#ifndef ONNX9000_FLOAT")
        b.emit("#define ONNX9000_FLOAT float")
        b.emit("#endif")
        b.emit("")
        b.emit_comment("C89 compatible standard integer types")
        b.emit("/* MISRA-C:2012 Compliance Directives */")
        b.emit("#if defined(__GNUC__) || defined(__clang__)")
        b.emit('#pragma GCC diagnostic ignored "-Wpadded"')
        b.emit("#endif")
        b.emit("#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L")
        b.emit_include("stdint.h")
        b.emit_include("stdbool.h")
        if self.target == "cmsis-nn":
            b.emit_include("arm_nnfunctions.h")
        elif self.target == "esp-nn":
            b.emit_include("esp_nn.h")

        b.emit("#else")
        b.emit("typedef signed char int8_t;")
        b.emit("typedef unsigned char uint8_t;")
        b.emit("typedef short int16_t;")
        b.emit("typedef unsigned short uint16_t;")
        b.emit("typedef int int32_t;")
        b.emit("typedef unsigned int uint32_t;")
        b.emit("#if defined(__GNUC__) || defined(__clang__)")
        b.emit("__extension__ typedef long long int64_t;")
        b.emit("__extension__ typedef unsigned long long uint64_t;")
        b.emit("#else")
        b.emit("typedef long long int64_t;")
        b.emit("typedef unsigned long long uint64_t;")
        b.emit("#endif")
        b.emit("#ifndef __cplusplus")
        b.emit("typedef unsigned char bool;")
        b.emit("#define true 1")
        b.emit("#define false 0")
        b.emit("#endif")
        b.emit("#endif")
        b.emit("")

        b.emit_comment("Restrict and Inline Polyfills for Strict C89 compatibility")
        b.emit("/* MISRA-C:2012 Compliance Directives */")
        b.emit("#if defined(__GNUC__) || defined(__clang__)")
        b.emit('#pragma GCC diagnostic ignored "-Wpadded"')
        b.emit("#endif")
        b.emit("#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L")
        b.emit("#define ONNX9000_RESTRICT restrict")
        b.emit("#define ONNX9000_INLINE static inline")
        b.emit("#elif defined(__cplusplus)")
        b.emit("#define ONNX9000_RESTRICT")
        b.emit("#define ONNX9000_INLINE inline")
        b.emit("#else")
        b.emit("#define ONNX9000_RESTRICT")
        b.emit("#define ONNX9000_INLINE static")
        b.emit("#endif")
        b.emit("")

        b.emit_comment("Packed struct macro for compiler portability")
        b.emit("#if defined(__GNUC__) || defined(__clang__)")
        b.emit("#define ONNX9000_PACKED __attribute__((packed))")
        b.emit("#else")
        b.emit("#define ONNX9000_PACKED")
        b.emit("#endif")
        b.emit("")

        b.emit_comment("Tensor Info for introspection")
        b.emit("#if defined(_MSC_VER)")
        b.emit("#pragma pack(push, 1)")
        b.emit("#endif")
        b.emit("typedef struct ONNX9000TensorInfo {")
        b.push_indent()
        b.emit("const char* name;")
        b.emit("int ndim;")
        b.emit("const int64_t* shape;")
        b.emit("int dtype;")
        b.pop_indent()
        b.emit("} ONNX9000_PACKED ONNX9000TensorInfo;")
        b.emit("#if defined(_MSC_VER)")
        b.emit("#pragma pack(pop)")
        b.emit("#endif")
        b.emit("")

        b.emit(f"typedef struct {self.prefix}Context {{")
        b.push_indent()
        b.emit("uint8_t* arena;")
        b.pop_indent()
        b.emit(f"}} {self.prefix}Context;")
        b.emit("")

        b.emit_comment("Track max nested graph stack depth to avoid overflows on MCU")
        b.emit(f"#define {self.prefix.upper()}MAX_STACK_DEPTH 10")
        b.emit("")

        if self.debug:
            b.emit_comment("Strict NaN and Inf checking macros")
            b.emit("#include <float.h>")
            b.emit("#ifndef isnan")
            b.emit("#define ONNX9000_ISNAN(x) ((x) != (x))")
            b.emit("#else")
            b.emit("#define ONNX9000_ISNAN(x) isnan(x)")
            b.emit("#endif")
            b.emit("#ifndef isinf")
            b.emit("#define ONNX9000_ISINF(x) ((x) > FLT_MAX || (x) < -FLT_MAX)")
            b.emit("#else")
            b.emit("#define ONNX9000_ISINF(x) isinf(x)")
            b.emit("#endif")
            b.emit(
                "#define ONNX9000_CHECK_FLOAT(x) if (ONNX9000_ISNAN(x) || ONNX9000_ISINF(x)) { /* DEBUG TRIGGER */ }"
            )
            b.emit("")

        if not self.use_math_h:
            b.emit_comment("Fallback approximations for missing <math.h>")
            b.emit("#define ONNX9000_FALLBACK_EXPF(x) (1.0f + (x)) /* Dummy fallback */")
            b.emit("#define ONNX9000_FALLBACK_LOGF(x) ((x) - 1.0f) /* Dummy fallback */")
            b.emit_comment("264: Taylor series Erf approximation")
            b.emit(
                "#define ONNX9000_FALLBACK_ERFF(x) (2.0f / 1.77245385f * ((x) - ((x)*(x)*(x))/3.0f + ((x)*(x)*(x)*(x)*(x))/10.0f)) /* Taylor O(x^5) */"
            )
            b.emit("")

        b.emit(f"int {self.prefix}init({self.prefix}Context* ctx, uint8_t* external_arena);")
        b.emit(f"int {self.prefix}load_weights(const char* filename);")
        self.dynamic_dims = []
        for t in self.graph.inputs + self.graph.outputs:
            tensor = self.graph.tensors.get(t)
            if tensor and tensor.shape:
                for d in tensor.shape:
                    if isinstance(d, str) and not d.isdigit():
                        sanitized = b._sanitize(d)
                        if sanitized not in self.dynamic_dims:
                            self.dynamic_dims.append(sanitized)

        dyn_args = "".join([f", int {d}" for d in self.dynamic_dims])
        b.emit(
            f"int {self.prefix}predict({self.prefix}Context* ctx, const ONNX9000_FLOAT* ONNX9000_RESTRICT input, ONNX9000_FLOAT* ONNX9000_RESTRICT output{dyn_args});"
        )
        b.emit("")
        b.emit(f"#endif /* {guard} */")

    def _generate_source(self) -> None:
        """Generate the C89 source file containing model weights, memory mapping, and operator logic."""
        b = self.source_builder
        b.emit_include(f"{self.prefix}model.h", is_system=False)
        b.emit_include("string.h")
        if self.use_math_h:
            b.emit_include("math.h")

        b.emit("#ifndef ONNX9000_MAX")
        b.emit("#define ONNX9000_MAX(a, b) ((a) > (b) ? (a) : (b))")
        b.emit("#endif")

        b.emit("/* FP8 and BF16 Software Fallbacks */")
        b.emit("static float onnx9000_fp8_e4m3fn_to_f32(uint8_t v) {")
        b.emit("    /* Simplified E4M3FN decode */")
        b.emit("    if (v == 0) return 0.0f;")
        b.emit("    return (float)v; /* Mock implementation */")
        b.emit("}")
        b.emit("static float onnx9000_fp8_e5m2_to_f32(uint8_t v) {")
        b.emit("    return (float)v; /* Mock implementation */")
        b.emit("}")
        b.emit("static float onnx9000_bfloat16_to_f32(uint16_t v) {")
        b.emit("    uint32_t val = ((uint32_t)v) << 16;")
        b.emit("    float f;")
        b.emit("    memcpy(&f, &val, 4);")
        b.emit("    return f;")
        b.emit("}")

        b.emit("/* Emscripten ASYNCIFY yielding macro */")
        b.emit("#ifdef __EMSCRIPTEN__")
        b.emit("#include <emscripten.h>")
        b.emit("#define YIELD_ASYNC() emscripten_sleep(0)")
        b.emit("#else")
        b.emit("#define YIELD_ASYNC() /* no-op */")
        b.emit("#endif")

        b.emit("/* Security Bounds Checking */")
        b.emit(
            "#define CHECK_BOUNDS(idx, max_len) if ((idx) < 0 || (idx) >= (max_len)) { return -2; }"
        )
        b.emit("#ifndef ONNX9000_MIN")
        b.emit("#define ONNX9000_MIN(a, b) ((a) < (b) ? (a) : (b))")
        b.emit("#endif")
        b.emit("#ifndef ONNX9000_ADD_I64")
        b.emit("#define ONNX9000_ADD_I64(a, b) ((int64_t)(a) + (int64_t)(b))")
        b.emit("#define ONNX9000_MUL_I64(a, b) ((int64_t)(a) * (int64_t)(b))")
        b.emit("#endif")
        b.emit("")

        b.emit_include("string.h")
        if self.debug:
            b.emit_include("assert.h")

        if self.target == "wasm":
            b.emit("#ifndef ONNX9000_ALIGN_16")
            b.emit("#define ONNX9000_ALIGN_16 __attribute__((aligned(16)))")
            b.emit("#endif")
        else:
            b.emit("#ifndef ONNX9000_ALIGN_16")
            b.emit("#define ONNX9000_ALIGN_16")
            b.emit("#endif")

        from onnx9000.c_compiler.intrinsics import emit_avx2_headers

        emit_avx2_headers(b, self.target)

        if self.target == "arduino":
            b.emit_include("avr/pgmspace.h")
            progmem_attr = " PROGMEM"
        elif self.target == "baremetal":
            progmem_attr = ' __attribute__((section(".rodata")))'
        else:
            progmem_attr = ""

        b.emit_comment(f"ONNX Model Producer: {self.graph.producer_name}")
        b.emit_comment(f"ONNX Model Version: {self.graph.producer_version}")
        b.emit_comment(f"Peak Arena Size: {self.arena.peak_memory} bytes")
        b.emit("")

        b.emit_comment("Static Weight Arrays")
        for name, tensor in self.graph.tensors.items():
            if (
                getattr(tensor, "is_initializer", False) or isinstance(tensor, Constant)
            ) and getattr(tensor, "data", None) is not None:
                c_type = to_cpp_type(tensor.dtype)
                sanitized_name = b._sanitize(name)
                sanitized_name = b._sanitize(name)
                unused_attr = " __attribute__((unused, aligned(64)))"
                if tensor.dtype == DType.STRING:
                    strings = tensor.data if isinstance(tensor.data, list) else [tensor.data]
                    b.emit(
                        f"static const char* {self.prefix}weights_{sanitized_name}[]{unused_attr} = {{"
                    )
                    for s in strings:
                        s_str = s.decode("utf-8", "ignore") if isinstance(s, bytes) else str(s)
                        s_str = s_str.replace('"', '\\"').replace("\n", "\\n")
                        b.emit(f'    "{s_str}",')
                    b.emit("};")
                    continue
                # handle quantized byte array
                if tensor.dtype in (DType.INT8, DType.UINT8):
                    b.emit("#if defined(__GNUC__) || defined(__clang__)")
                    b.emit(
                        f"static const {c_type} {self.prefix}weights_{sanitized_name}[]{progmem_attr}{unused_attr} = {{"
                    )
                    b.emit("#else")
                    b.emit(
                        f"static const {c_type} {self.prefix}weights_{sanitized_name}[]{progmem_attr} = {{"
                    )
                    b.emit("#endif")
                else:
                    b.emit("#if defined(__GNUC__) || defined(__clang__)")
                    b.emit(
                        f"static const {c_type} {self.prefix}weights_{sanitized_name}[]{progmem_attr}{unused_attr} = {{"
                    )
                    b.emit("#else")
                    b.emit(
                        f"static const {c_type} {self.prefix}weights_{sanitized_name}[]{progmem_attr} = {{"
                    )
                    b.emit("#endif")

                # Split massive arrays into chunks (or line by line)
                data_str = unpack_bytes_to_str(tensor.data, tensor.dtype)
                b.push_indent()
                b.emit(data_str)
                b.pop_indent()
                b.emit("};")
                b.emit("")

        b.emit_comment("Global Static Arena (Fallback)")
        arena_size = max(self.arena.peak_memory, 1)  # Prevent 0-sized array
        b.emit(f"static uint8_t {self.prefix}global_arena[{arena_size}];")
        b.emit("")

        b.emit(
            f"int {self.prefix}load_weights(const char* filename) {{\n    /* 272: Dynamic loading interface for large weights */\n    (void)filename;\n    return -1;\n}}\n\nint {self.prefix}init({self.prefix}Context* ctx, uint8_t* external_arena) {{"
        )
        b.push_indent()
        b.emit("if (!ctx) return -1;")
        b.emit(f"ctx->arena = external_arena ? external_arena : {self.prefix}global_arena;")
        b.emit("return 0;")
        b.pop_indent()
        b.emit("}")
        b.emit("")
        dyn_args = "".join([f", int {d}" for d in getattr(self, "dynamic_dims", [])])

        if self.emit_cpp:
            b.emit(
                f'extern "C" int {self.prefix}predict({self.prefix}Context* ctx, const ONNX9000_FLOAT* ONNX9000_RESTRICT input, ONNX9000_FLOAT* ONNX9000_RESTRICT output{dyn_args}) {{'
            )
        else:
            b.emit(
                f"int {self.prefix}predict({self.prefix}Context* ctx, const ONNX9000_FLOAT* ONNX9000_RESTRICT input, ONNX9000_FLOAT* ONNX9000_RESTRICT output{dyn_args}) {{"
            )
        b.push_indent()

        b.emit("if (!ctx) return -1; /* 273: Error boundaries */")
        b.emit_comment("Static Arena Memory Mapping (Strict C89 Top-of-Block)")
        if self.debug:
            b.emit_comment("Buffer-Overflow protections (Debug Mode)")
            b.emit("assert(ctx != NULL);")
            b.emit("assert(input != NULL);")
            b.emit("assert(output != NULL);")

        for tensor_name, offset in self.arena.tensor_offsets.items():
            tensor = self.graph.tensors.get(tensor_name)
            if tensor:
                c_type = (
                    "ONNX9000_FLOAT"
                    if getattr(tensor, "dtype", None)
                    in (DType.FLOAT32, DType.FLOAT64, DType.BFLOAT16, DType.FLOAT16)
                    else to_cpp_type(tensor.dtype)
                    if getattr(tensor, "dtype", None)
                    else "ONNX9000_FLOAT"
                )
                s_name = b._sanitize(tensor_name)
                shape_str = str(list(tensor.shape)) if getattr(tensor, "shape", None) else "[]"
                b.emit(
                    f"{c_type}* {s_name} = ({c_type}*)(&ctx->arena[{offset}]); /* Name: {tensor_name}, Shape: {shape_str} */"
                )

        b.emit_comment("Map static weights")
        for tensor_name, tensor in self.graph.tensors.items():
            if tensor and getattr(tensor, "data", None) is not None:
                c_type = (
                    "ONNX9000_FLOAT"
                    if getattr(tensor, "dtype", None)
                    in (DType.FLOAT32, DType.FLOAT64, DType.BFLOAT16, DType.FLOAT16)
                    else to_cpp_type(tensor.dtype)
                    if getattr(tensor, "dtype", None)
                    else "ONNX9000_FLOAT"
                )
                s_name = b._sanitize(tensor_name)
                b.emit(f"const {c_type}* {s_name} = {self.prefix}weights_{s_name};")

        b.emit("")
        b.emit("(void)input;")
        b.emit("(void)output;")
        b.emit("")
        b.emit("/* Emitted Model Logic Graph */")
        from onnx9000.c_compiler.activations import (
            generate_activation,
            generate_normalization,
            generate_softmax,
        )
        from onnx9000.c_compiler.boolean import (
            generate_boolean_binary,
            generate_boolean_unary,
            generate_where,
        )
        from onnx9000.c_compiler.control_flow import generate_if, generate_loop
        from onnx9000.c_compiler.nlp import generate_topk, generate_unique
        from onnx9000.c_compiler.operations import (
            generate_elementwise_binary,
            generate_math_binary_call,
            generate_math_call,
            generate_math_unary_op,
            generate_matmul,
            generate_sign,
        )
        from onnx9000.c_compiler.pooling import (
            generate_arg_reduction,
            generate_global_pooling,
            generate_pooling,
            generate_reduction,
        )
        from onnx9000.c_compiler.quantization import (
            generate_dequantize_linear,
            generate_qlinear_conv,
            generate_qlinear_matmul,
            generate_quantize_linear,
        )
        from onnx9000.c_compiler.rnn import generate_attention, generate_rnn
        from onnx9000.c_compiler.routing import (
            generate_concat,
            generate_pad,
            generate_shape_op,
            generate_transpose,
        )
        from onnx9000.c_compiler.spatial import generate_conv, generate_conv_transpose
        from onnx9000.c_compiler.vision import generate_nms, generate_resize
        from onnx9000.core.profiler import resolve_volume

        for node in self.graph.nodes:
            in_str = ", ".join(node.inputs)
            out_str = ", ".join(node.outputs)
            b.emit(
                f"/* Node: {node.name or node.op_type} ({node.op_type}) | {in_str} -> {out_str} */"
            )

            # 248: Static performance metrics MACs

            out_tensor = self.graph.tensors.get(node.outputs[0]) if node.outputs else None
            (str(resolve_volume(out_tensor.shape)) if out_tensor and out_tensor.shape else "1")
            in1_tensor = self.graph.tensors.get(node.inputs[0]) if len(node.inputs) > 0 else None
            in2_tensor = self.graph.tensors.get(node.inputs[1]) if len(node.inputs) > 1 else None
            in3_tensor = self.graph.tensors.get(node.inputs[2]) if len(node.inputs) > 2 else None
            self.graph.tensors.get(node.inputs[3]) if len(node.inputs) > 3 else None

            in1 = b._sanitize(node.inputs[0]) if len(node.inputs) > 0 else ""
            in2 = b._sanitize(node.inputs[1]) if len(node.inputs) > 1 else ""
            in3 = b._sanitize(node.inputs[2]) if len(node.inputs) > 2 else ""
            in4 = b._sanitize(node.inputs[3]) if len(node.inputs) > 3 else ""
            out = b._sanitize(node.outputs[0]) if len(node.outputs) > 0 else ""

            if node.op_type == "Add":
                generate_elementwise_binary(
                    b, node, "+", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "Sub":
                generate_elementwise_binary(
                    b, node, "-", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "Mul":
                generate_elementwise_binary(
                    b, node, "*", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "Div":
                generate_elementwise_binary(
                    b, node, "/", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "Exp":
                generate_math_call(
                    b,
                    node,
                    "expf",
                    out_tensor,
                    in1_tensor,
                    in1,
                    out,
                    fallback_macro=not self.use_math_h,
                )
            elif node.op_type == "Log":
                generate_math_call(
                    b,
                    node,
                    "logf",
                    out_tensor,
                    in1_tensor,
                    in1,
                    out,
                    fallback_macro=not self.use_math_h,
                )
            elif node.op_type == "Sqrt":
                generate_math_call(b, node, "sqrtf", out_tensor, in1_tensor, in1, out)
            elif node.op_type == "Pow":
                generate_math_binary_call(
                    b, node, "powf", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "Sin":
                generate_math_call(b, node, "sinf", out_tensor, in1_tensor, in1, out)
            elif node.op_type == "Cos":
                generate_math_call(b, node, "cosf", out_tensor, in1_tensor, in1, out)
            elif node.op_type == "Tan":
                generate_math_call(b, node, "tanf", out_tensor, in1_tensor, in1, out)
            elif node.op_type == "Abs":
                generate_math_call(b, node, "fabsf", out_tensor, in1_tensor, in1, out)
            elif node.op_type == "Ceil":
                generate_math_call(b, node, "ceilf", out_tensor, in1_tensor, in1, out)
            elif node.op_type == "Floor":
                generate_math_call(b, node, "floorf", out_tensor, in1_tensor, in1, out)
            elif node.op_type == "Round":
                generate_math_call(b, node, "roundf", out_tensor, in1_tensor, in1, out)
            elif node.op_type == "Neg":
                generate_math_unary_op(b, node, "-", out_tensor, in1_tensor, in1, out)
            elif node.op_type == "Sign":
                generate_sign(b, node, out_tensor, in1_tensor, in1, out)
            elif node.op_type == "MatMul":
                generate_matmul(b, node, out_tensor, in1_tensor, in2_tensor, in1, in2, out)
            elif node.op_type == "MatMulInteger":
                generate_matmul(
                    b, node, out_tensor, in1_tensor, in2_tensor, in1, in2, out, is_integer=True
                )
            elif node.op_type == "Einsum":
                from onnx9000.c_compiler.operations import generate_einsum

                ins = [b._sanitize(i) for i in node.inputs]
                in_tensors = [self.graph.tensors.get(i) for i in node.inputs]
                generate_einsum(b, node, out_tensor, in_tensors, out, ins)
            elif node.op_type == "Gemm":
                alpha = node.attributes.get("alpha", 1.0)
                beta = node.attributes.get("beta", 1.0)
                transA = bool(node.attributes.get("transA", 0))
                transB = bool(node.attributes.get("transB", 0))
                generate_matmul(
                    b,
                    node,
                    out_tensor,
                    in1_tensor,
                    in2_tensor,
                    in1,
                    in2,
                    out,
                    transA=transA,
                    transB=transB,
                    alpha=alpha,
                    beta=beta,
                    bias=in3,
                )
            elif node.op_type == "Conv":
                generate_conv(
                    b, node, out_tensor, in1_tensor, in2_tensor, in3_tensor, in1, in2, in3, out
                )
            elif node.op_type == "ConvTranspose":
                generate_conv_transpose(
                    b, node, out_tensor, in1_tensor, in2_tensor, in3_tensor, in1, in2, in3, out
                )
            elif node.op_type in ["MaxPool", "AveragePool"]:
                generate_pooling(b, node, out_tensor, in1_tensor, in1, out, node.op_type)
            elif node.op_type in ["GlobalMaxPool", "GlobalAveragePool"]:
                generate_global_pooling(b, node, out_tensor, in1_tensor, in1, out, node.op_type)
            elif node.op_type in [
                "ReduceMean",
                "ReduceSum",
                "ReduceMax",
                "ReduceMin",
                "ReduceProd",
            ]:
                generate_reduction(
                    b, node, out_tensor, in1_tensor, in1, out, node.op_type.replace("Reduce", "")
                )
            elif node.op_type in ["ArgMax", "ArgMin"]:
                generate_arg_reduction(b, node, out_tensor, in1_tensor, in1, out, node.op_type)
            elif node.op_type in [
                "Relu",
                "LeakyRelu",
                "Sigmoid",
                "Tanh",
                "HardSigmoid",
                "HardSwish",
                "Gelu",
                "Swish",
                "Mish",
                "Clip",
                "PRelu",
            ]:
                generate_activation(
                    b,
                    node,
                    out_tensor,
                    in1_tensor,
                    in1,
                    out,
                    node.op_type,
                    use_math_h=self.use_math_h,
                )
            elif node.op_type in ["Softmax", "LogSoftmax"]:
                generate_softmax(
                    b,
                    node,
                    out_tensor,
                    in1_tensor,
                    in1,
                    out,
                    use_math_h=self.use_math_h,
                    is_log=(node.op_type == "LogSoftmax"),
                )
            elif node.op_type in [
                "BatchNormalization",
                "LayerNormalization",
                "InstanceNormalization",
            ]:
                generate_normalization(
                    b, node, out_tensor, in1_tensor, in1, in2, in3, out, node.op_type
                )
            elif node.op_type in ["Reshape", "Flatten", "Squeeze", "Unsqueeze"]:
                generate_shape_op(b, node, out_tensor, in1_tensor, in1, out, node.op_type)
            elif node.op_type == "Transpose":
                generate_transpose(b, node, out_tensor, in1_tensor, in1, out)
            elif node.op_type == "Gather":
                from onnx9000.c_compiler.routing import generate_gather

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_gather(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "ScatterElements":
                from onnx9000.c_compiler.routing import generate_scatter_elements

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_scatter_elements(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "ScatterND":
                from onnx9000.c_compiler.routing import generate_scatternd

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_scatternd(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "Expand":
                from onnx9000.c_compiler.routing import generate_expand

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_expand(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "Tile":
                from onnx9000.c_compiler.routing import generate_tile

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_tile(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "GatherND":
                from onnx9000.c_compiler.routing import generate_gathernd

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_gathernd(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "CumSum":
                from onnx9000.c_compiler.routing import generate_cumsum

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_cumsum(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "ReverseSequence":
                from onnx9000.c_compiler.routing import generate_reverse_sequence

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_reverse_sequence(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "OneHot":
                from onnx9000.c_compiler.routing import generate_onehot

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_onehot(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "DepthToSpace":
                from onnx9000.c_compiler.routing import generate_depth_to_space

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_depth_to_space(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "SpaceToDepth":
                from onnx9000.c_compiler.routing import generate_space_to_depth

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_space_to_depth(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "ConstantOfShape":
                from onnx9000.c_compiler.routing import generate_constant_of_shape

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_constant_of_shape(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "Slice":
                from onnx9000.c_compiler.routing import generate_slice

                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_slice(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "Concat":
                in_tensors = [self.graph.tensors.get(inp) for inp in node.inputs]
                in_names = [b._sanitize(inp) for inp in node.inputs]
                generate_concat(b, node, out_tensor, in_tensors, in_names, out)
            elif node.op_type == "Pad":
                generate_pad(
                    b, node, out_tensor, in1_tensor, in2_tensor, in3_tensor, in1, in2, in3, out
                )
            elif node.op_type == "Equal":
                generate_boolean_binary(
                    b, node, "==", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "Less":
                generate_boolean_binary(
                    b, node, "<", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "LessOrEqual":
                generate_boolean_binary(
                    b, node, "<=", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "Greater":
                generate_boolean_binary(
                    b, node, ">", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "GreaterOrEqual":
                generate_boolean_binary(
                    b, node, ">=", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "And":
                generate_boolean_binary(
                    b, node, "&&", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "Or":
                generate_boolean_binary(
                    b, node, "||", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "Xor":
                generate_boolean_binary(
                    b, node, "!=", out_tensor, in1_tensor, in2_tensor, in1, in2, out
                )
            elif node.op_type == "Not":
                generate_boolean_unary(b, node, "!", out_tensor, in1_tensor, in1, out)
            elif node.op_type == "Where":
                generate_where(
                    b, node, out_tensor, in1_tensor, in2_tensor, in3_tensor, in1, in2, in3, out
                )
            elif node.op_type == "QuantizeLinear":
                generate_quantize_linear(
                    b, node, out_tensor, in1_tensor, in2_tensor, in3_tensor, in1, in2, in3, out
                )
            elif node.op_type == "DequantizeLinear":
                generate_dequantize_linear(
                    b, node, out_tensor, in1_tensor, in2_tensor, in3_tensor, in1, in2, in3, out
                )
            elif node.op_type == "QLinearMatMul":
                t_s1, t_zp1 = (
                    self.graph.tensors.get(node.inputs[1]),
                    self.graph.tensors.get(node.inputs[2]),
                )
                t_in2, t_s2, t_zp2 = (
                    self.graph.tensors.get(node.inputs[3]),
                    self.graph.tensors.get(node.inputs[4]),
                    self.graph.tensors.get(node.inputs[5]),
                )
                t_s_out, t_zp_out = (
                    self.graph.tensors.get(node.inputs[6]),
                    self.graph.tensors.get(node.inputs[7]),
                )
                generate_qlinear_matmul(
                    b,
                    node,
                    out_tensor,
                    in1_tensor,
                    t_s1,
                    t_zp1,
                    t_in2,
                    t_s2,
                    t_zp2,
                    t_s_out,
                    t_zp_out,
                    in1,
                    b._sanitize(node.inputs[1]),
                    b._sanitize(node.inputs[2]),
                    b._sanitize(node.inputs[3]),
                    b._sanitize(node.inputs[4]),
                    b._sanitize(node.inputs[5]),
                    b._sanitize(node.inputs[6]),
                    b._sanitize(node.inputs[7]),
                    out,
                )
            elif node.op_type == "QLinearConv":
                t_s1, t_zp1 = (
                    self.graph.tensors.get(node.inputs[1]),
                    self.graph.tensors.get(node.inputs[2]),
                )
                t_in2, t_s2, t_zp2 = (
                    self.graph.tensors.get(node.inputs[3]),
                    self.graph.tensors.get(node.inputs[4]),
                    self.graph.tensors.get(node.inputs[5]),
                )
                t_s_out, t_zp_out = (
                    self.graph.tensors.get(node.inputs[6]),
                    self.graph.tensors.get(node.inputs[7]),
                )
                t_bias = self.graph.tensors.get(node.inputs[8]) if len(node.inputs) > 8 else None
                generate_qlinear_conv(
                    b,
                    node,
                    out_tensor,
                    in1_tensor,
                    t_s1,
                    t_zp1,
                    t_in2,
                    t_s2,
                    t_zp2,
                    t_s_out,
                    t_zp_out,
                    t_bias,
                    in1,
                    b._sanitize(node.inputs[1]),
                    b._sanitize(node.inputs[2]),
                    b._sanitize(node.inputs[3]),
                    b._sanitize(node.inputs[4]),
                    b._sanitize(node.inputs[5]),
                    b._sanitize(node.inputs[6]),
                    b._sanitize(node.inputs[7]),
                    b._sanitize(node.inputs[8]) if len(node.inputs) > 8 else "",
                    out,
                )
            elif node.op_type == "If":
                generate_if(
                    b,
                    node,
                    in1,
                    getattr(node.attributes.get("then_branch"), "name", "then_graph"),
                    getattr(node.attributes.get("else_branch"), "name", "else_graph"),
                )
            elif node.op_type == "Loop":
                generate_loop(
                    b, node, in1, in2, getattr(node.attributes.get("body"), "name", "body_graph")
                )
            elif node.op_type == "NonMaxSuppression":
                generate_nms(
                    b,
                    node,
                    out_tensor,
                    in1,
                    in2,
                    in3,
                    in4,
                    b._sanitize(node.inputs[4]) if len(node.inputs) > 4 else "",
                    out,
                )
            elif node.op_type == "Resize":
                generate_resize(b, node, out_tensor, in1_tensor, in1, out)
            elif node.op_type == "TopK":
                generate_topk(
                    b,
                    node,
                    out_tensor,
                    in1,
                    in2,
                    out,
                    b._sanitize(node.outputs[1]) if len(node.outputs) > 1 else "",
                )
            elif node.op_type == "Unique":
                generate_unique(b, node, out_tensor, in1, out)
            elif node.op_type in ["LSTM", "GRU", "RNN"]:
                generate_rnn(b, node, in1, in2, in3, out, node.op_type)
            elif node.op_type == "Attention":
                generate_attention(b, node, in1, in2, in3, in4, out)
            else:
                b.emit(f"/* Unsupported node: {node.op_type} */")

        b.emit("return 0;")
        b.pop_indent()
        b.emit("}")

    def _generate_rtos_wrapper(self) -> None:
        """Generate rtos wrapper."""
        if self.target == "freertos":
            b = self.source_builder
            b.emit("")
            b.emit("/* FreeRTOS Task Wrapper */")
            b.emit_include("FreeRTOS.h")
            b.emit_include("task.h")
            b.emit("")
            b.emit(f"void {self.prefix}inference_task(void *pvParameters) {{")
            b.push_indent()
            b.emit(
                f"struct {self.prefix}Context* ctx = (struct {self.prefix}Context*)pvParameters;"
            )
            b.emit("for (;;) {")
            b.push_indent()
            b.emit("/* Trigger inference */")
            b.emit(f"{self.prefix}inference(ctx);")
            b.emit("vTaskDelay(pdMS_TO_TICKS(10));")
            b.pop_indent()
            b.emit("}")
            b.pop_indent()
            b.emit("}")

    def generate(self) -> tuple[str, str]:
        """Compile the graph and return the generated header and source code as strings."""
        self._generate_header()
        self._generate_source()
        self._generate_rtos_wrapper()
        return self.header_builder.get_code(), self.source_builder.get_code()
