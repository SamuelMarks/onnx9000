/**
 * WGSL Shader for Keras LayerNormalization.
 * @returns The WGSL shader source code.
 */
export const getKerasLayerNormWGSL = () => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Uniforms {
    batchSize: u32,
    seqLen: u32,
    hiddenSize: u32,
    epsilon: f32,
};
@group(0) @binding(4) var<uniform> config: Uniforms;

var<workgroup> mean_shared: array<f32, 256>;
var<workgroup> var_shared: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_seq_idx = workgroup_id.x;
    let local_idx = local_id.x;
    
    if (batch_seq_idx >= config.batchSize * config.seqLen) {
        return;
    }
    
    var sum: f32 = 0.0;
    var sq_sum: f32 = 0.0;
    
    let base_idx = batch_seq_idx * config.hiddenSize;
    
    // Parallel reduction step 1
    for (var i: u32 = local_idx; i < config.hiddenSize; i = i + 256u) {
        let val = input[base_idx + i];
        sum = sum + val;
        sq_sum = sq_sum + val * val;
    }
    
    mean_shared[local_idx] = sum;
    var_shared[local_idx] = sq_sum;
    workgroupBarrier();
    
    // Parallel reduction step 2
    for (var offset: u32 = 128u; offset > 0u; offset = offset / 2u) {
        if (local_idx < offset) {
            mean_shared[local_idx] = mean_shared[local_idx] + mean_shared[local_idx + offset];
            var_shared[local_idx] = var_shared[local_idx] + var_shared[local_idx + offset];
        }
        workgroupBarrier();
    }
    
    let mean = mean_shared[0] / f32(config.hiddenSize);
    let variance = (var_shared[0] / f32(config.hiddenSize)) - (mean * mean);
    let inv_std = inverseSqrt(variance + config.epsilon);
    
    // Apply normalization
    for (var i: u32 = local_idx; i < config.hiddenSize; i = i + 256u) {
        let val = input[base_idx + i];
        let normalized = (val - mean) * inv_std;
        output[base_idx + i] = normalized * gamma[i] + beta[i];
    }
}
`;
