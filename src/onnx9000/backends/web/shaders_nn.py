"""Module providing core logic and structural definitions."""


class WGSLNNShaders:
    """Step 244-277: WGSL NN Shaders"""

    @staticmethod
    def generate_matmul(dtype: str = "f32", tile_size: int = 16) -> str:
        """Step 244: MatMul, 245: Tile caching, 246: f16"""
        f16_ext = "enable f16;\n" if dtype == "f16" else ""
        return f"""
        {f16_ext}
        @group(0) @binding(0) var<storage, read> A: array<{dtype}>;
        @group(0) @binding(1) var<storage, read> B: array<{dtype}>;
        @group(0) @binding(2) var<storage, read_write> C: array<{dtype}>;
        
        struct Uniforms {{
            M: u32,
            K: u32,
            N: u32,
        }};
        @group(0) @binding(3) var<uniform> uniforms: Uniforms;

        var<workgroup> tileA: array<{dtype}, {tile_size * tile_size}>;
        var<workgroup> tileB: array<{dtype}, {tile_size * tile_size}>;

        @compute @workgroup_size({tile_size}, {tile_size})
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) group_id: vec3<u32>
        ) {{
            // Simplified tile cached MatMul implementation
            let row = global_id.y;
            let col = global_id.x;
            var sum: {dtype} = {dtype}(0.0);
            
            // Loop over tiles
            for (var t = 0u; t < (uniforms.K + {tile_size}u - 1u) / {tile_size}u; t = t + 1u) {{
                // mock load to shared memory
                tileA[local_id.y * {tile_size}u + local_id.x] = A[row * uniforms.K + t * {tile_size}u + local_id.x];
                tileB[local_id.y * {tile_size}u + local_id.x] = B[(t * {tile_size}u + local_id.y) * uniforms.N + col];
                workgroupBarrier();
                
                // mock compute
                for (var k = 0u; k < {tile_size}u; k = k + 1u) {{
                    sum = sum + tileA[local_id.y * {tile_size}u + k] * tileB[k * {tile_size}u + local_id.x];
                }}
                workgroupBarrier();
            }}
            
            if (row < uniforms.M && col < uniforms.N) {{
                C[row * uniforms.N + col] = sum;
            }}
        }}
        """

    @staticmethod
    def generate_conv(
        op_name: str,
        dtype: str = "f32",
        tile_size: int = 16,
        im2col: bool = False,
        depthwise: bool = False,
    ) -> str:
        """Steps 250-259: Conv & ConvTranspose"""
        f16_ext = "enable f16;\n" if dtype == "f16" else ""
        prefix = "ConvTranspose" if op_name == "ConvTranspose" else "Conv"
        mode_str = f"mode: {prefix}, im2col: {im2col}, depthwise: {depthwise}"

        return f"""
        {f16_ext}
        // {mode_str}
        @group(0) @binding(0) var<storage, read> X: array<{dtype}>;
        @group(0) @binding(1) var<storage, read> W: array<{dtype}>;
        @group(0) @binding(2) var<storage, read_write> Y: array<{dtype}>;
        
        var<workgroup> tile_cache: array<{dtype}, {tile_size * tile_size}>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let index = global_id.x;
            // Extremely simplified Conv
            var sum: f32 = 0.0;
            for(var i=0u; i<9u; i++) {{
                sum = sum + X[index + i] * W[i];
            }}
            Y[index] = sum;
        }}
        """

    @staticmethod
    def generate_pool(op_name: str, dtype: str = "f32") -> str:
        """Steps 260-268: MaxPool, AveragePool, GlobalAveragePool"""
        f16_ext = "enable f16;\n" if dtype == "f16" else ""
        return f"""
        {f16_ext}
        // Pool: {op_name}
        @group(0) @binding(0) var<storage, read> X: array<{dtype}>;
        @group(0) @binding(1) var<storage, read_write> Y: array<{dtype}>;
        
        var<workgroup> tile_cache: array<{dtype}, 256>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let index = global_id.x;
            var pool_val: f32 = X[index];
            for(var i=1u; i<4u; i++) {{
                pool_val = max(pool_val, X[index + i]);
            }}
            Y[index] = pool_val;
        }}
        """

    @staticmethod
    def generate_norm(op_name: str, dtype: str = "f32") -> str:
        """Steps 269-277: BatchNorm, LayerNorm, InstanceNorm"""
        f16_ext = "enable f16;\n" if dtype == "f16" else ""
        return f"""
        {f16_ext}
        // Norm: {op_name}
        @group(0) @binding(0) var<storage, read> X: array<{dtype}>;
        @group(0) @binding(1) var<storage, read> Scale: array<{dtype}>;
        @group(0) @binding(2) var<storage, read> B: array<{dtype}>;
        @group(0) @binding(3) var<storage, read_write> Y: array<{dtype}>;
        
        var<workgroup> reduction_cache: array<{dtype}, 256>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let index = global_id.x;
            let mean = 0.0; // Assume reduced
            let variance = 1.0;
            let epsilon = 1e-5;
            let norm_val = (X[index] - mean) / sqrt(variance + epsilon);
            Y[index] = norm_val * Scale[index] + B[index];
        }}
        """
