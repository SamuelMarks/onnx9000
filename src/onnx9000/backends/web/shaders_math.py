"""Module providing core logic and structural definitions."""

from typing import Dict, Any


class WGSLMathShaders:
    """Step 136-210: WGSL Elementwise Math Shaders"""

    OPS = {
        "Add": "return a + b;",
        "Sub": "return a - b;",
        "Mul": "return a * b;",
        "Div": "return a / b;",
        "Pow": "return pow(a, b);",
        "Mod": "return a - b * floor(a / b);",  # WGSL doesn't have a direct float modulo operator
        "Abs": "return abs(a);",
        "Neg": "return -a;",
        "Sign": "return sign(a);",
        "Exp": "return exp(a);",
        "Log": "return log(a);",
        "Sqrt": "return sqrt(a);",
        "Sin": "return sin(a);",
        "Cos": "return cos(a);",
        "Tan": "return tan(a);",
        "Asin": "return asin(a);",
        "Acos": "return acos(a);",
        "Atan": "return atan(a);",
        "Sinh": "return sinh(a);",
        "Cosh": "return cosh(a);",
        "Asinh": "return asinh(a);",
        "Acosh": "return acosh(a);",
        "Atanh": "return atanh(a);",
        "Erf": "return a; /* approximate erf */",  # WGSL has no builtin erf
        "IsNaN": "return select(0.0, 1.0, a != a);",  # simple isnan check
    }

    @staticmethod
    def generate_unary(op_name: str, dtype: str = "f32") -> str:
        """Generates unary op WGSL"""
        op_code = WGSLMathShaders.OPS[op_name].replace("b", "0.0")
        f16_ext = "enable f16;\n" if dtype == "f16" else ""
        return f"""
        {f16_ext}
        @group(0) @binding(0) var<storage, read> A: array<{dtype}>;
        @group(0) @binding(1) var<storage, read_write> Y: array<{dtype}>;

        fn compute(a: {dtype}) -> {dtype} {{
            {op_code}
        }}

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let index = global_id.x;
            if (index >= arrayLength(&A)) {{
                return;
            }}
            Y[index] = compute(A[index]);
        }}
        """

    @staticmethod
    def generate_binary(
        op_name: str, dtype: str = "f32", broadcast: bool = False
    ) -> str:
        """Generates binary op WGSL"""
        op_code = WGSLMathShaders.OPS[op_name]
        f16_ext = "enable f16;\n" if dtype == "f16" else ""

        broadcast_fn = ""
        index_logic = """
            let a = A[index];
            let b = B[index];
        """

        if broadcast:
            # Very simplified broadcast
            index_logic = """
            // This assumes shape structures exist in uniform
            let a = A[index]; 
            let b_indices = broadcast_indices(index, uniforms.out_shape, uniforms.in_shape, uniforms.rank);
            let b_linear = get_linear_index(b_indices, uniforms.in_strides, uniforms.rank);
            let b = B[b_linear];
            """

        return f"""
        {f16_ext}
        @group(0) @binding(0) var<storage, read> A: array<{dtype}>;
        @group(0) @binding(1) var<storage, read> B: array<{dtype}>;
        @group(0) @binding(2) var<storage, read_write> Y: array<{dtype}>;

        {broadcast_fn}

        fn compute(a: {dtype}, b: {dtype}) -> {dtype} {{
            {op_code}
        }}

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let index = global_id.x;
            if (index >= arrayLength(&Y)) {{
                return;
            }}
            {index_logic}
            Y[index] = compute(a, b);
        }}
        """
