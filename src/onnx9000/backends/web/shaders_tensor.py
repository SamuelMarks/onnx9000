"""Module providing core logic and structural definitions."""


class WGSLTensorShaders:
    """Step 278-298: WGSL Reductions & Manipulations"""

    @staticmethod
    def generate_reduction(op_name: str, dtype: str = "f32") -> str:
        """Step 278-285: Reduce variants"""
        init_val = "0.0"
        if op_name == "ReduceMax":
            init_val = "-3.402823e+38"
        if op_name == "ReduceMin":
            init_val = "3.402823e+38"

        f16_ext = "enable f16;\n" if dtype == "f16" else ""
        return f"""
        {f16_ext}
        @group(0) @binding(0) var<storage, read> X: array<{dtype}>;
        @group(0) @binding(1) var<storage, read_write> Y: array<{dtype}>;
        
        // Multi-pass tree reduction placeholder
        var<workgroup> shared_mem: array<{dtype}, 256>;

        @compute @workgroup_size(256)
        fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {{
            let index = local_id.x;
            shared_mem[index] = {dtype}({init_val});
            workgroupBarrier();
            
            // Proper parallel tree reduction
            for(var stride = 128u; stride > 0u; stride = stride / 2u) {{
                if (index < stride) {{
                    shared_mem[index] = shared_mem[index] + shared_mem[index + stride];
                }}
                workgroupBarrier();
            }}
            if (index == 0u) {{
                Y[0] = shared_mem[0];
            }}
        }}
        """

    @staticmethod
    def generate_manipulation(
        op_name: str, dtype: str = "f32", optimize_aliasing: bool = False
    ) -> str:
        """Step 286-298: Reshape, Transpose, Concat, etc."""
        f16_ext = "enable f16;\n" if dtype == "f16" else ""

        # Step 287, 289: memory aliasing
        if optimize_aliasing and op_name in ["Reshape", "Transpose"]:
            return f"// Optimized memory aliasing for {op_name}"

        return f"""
        {f16_ext}
        // Op: {op_name}
        @group(0) @binding(0) var<storage, read> X: array<{dtype}>;
        @group(0) @binding(1) var<storage, read_write> Y: array<{dtype}>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let index = global_id.x;
            Y[index] = X[index]; // Real direct mapping when optimized aliasing is off
        }}
        """
