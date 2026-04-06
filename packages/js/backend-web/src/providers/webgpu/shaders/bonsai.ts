export const FusedLayerNormWGSL = `
struct LayerNormUniforms {
    N: u32,
    epsilon: f32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> gamma : array<f32>;
@group(0) @binding(2) var<storage, read> beta : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;
@group(0) @binding(4) var<uniform> uniforms : LayerNormUniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let N = uniforms.N;
    let row = global_id.x;
    
    // Simplistic fused layer norm
    var sum = 0.0;
    for (var i = 0u; i < N; i = i + 1u) {
        sum = sum + input[row * N + i];
    }
    let mean = sum / f32(N);
    
    var variance = 0.0;
    for (var i = 0u; i < N; i = i + 1u) {
        let diff = input[row * N + i] - mean;
        variance = variance + diff * diff;
    }
    variance = variance / f32(N);
    let inv_std = 1.0 / sqrt(variance + uniforms.epsilon);
    
    for (var i = 0u; i < N; i = i + 1u) {
        let norm_val = (input[row * N + i] - mean) * inv_std;
        output[row * N + i] = norm_val * gamma[i] + beta[i];
    }
}
`;

export const FusedRoPEWGSL = `
struct RoPEUniforms {
    seq_len: u32,
    head_dim: u32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> cos_cache : array<f32>;
@group(0) @binding(2) var<storage, read> sin_cache : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;
@group(0) @binding(4) var<uniform> uniforms : RoPEUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let seq_idx = global_id.x;
    let head_idx = global_id.y;
    let dim_idx = global_id.z * 2u;
    
    if (dim_idx >= uniforms.head_dim) {
        return;
    }
    
    let base_idx = seq_idx * uniforms.head_dim + dim_idx;
    let v0 = input[base_idx];
    let v1 = input[base_idx + 1u];
    
    let cos_val = cos_cache[seq_idx * uniforms.head_dim + dim_idx];
    let sin_val = sin_cache[seq_idx * uniforms.head_dim + dim_idx];
    
    output[base_idx] = v0 * cos_val - v1 * sin_val;
    output[base_idx + 1u] = v0 * sin_val + v1 * cos_val;
}
`;
