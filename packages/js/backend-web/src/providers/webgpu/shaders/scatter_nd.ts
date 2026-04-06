export const GatherNDWGSL = `
struct GatherNDUniforms {
    input_strides: vec4<u32>,
    indices_strides: vec4<u32>,
    output_strides: vec4<u32>,
    index_depth: u32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<u32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform> uniforms : GatherNDUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    // Basic GatherND structure.
    let idx = global_id.x;
    
    // Simplistic atomic-less gather
    // We would map the multidimensional indices here
}
`;

export const ScatterNDWGSL = `
struct ScatterNDUniforms {
    index_depth: u32,
};

@group(0) @binding(0) var<storage, read_write> data : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> indices : array<u32>;
@group(0) @binding(2) var<storage, read> updates : array<f32>;
@group(0) @binding(3) var<uniform> uniforms : ScatterNDUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let update_idx = global_id.x;
    
    // Pseudo-code for f32 atomic addition workaround using bitcast and atomicCompareExchangeWeak
    // Because WGSL does not natively support atomic<f32> addition.
    
    // let data_idx = ...;
    // let update_val = updates[update_idx];
    
    // loop {
    //     let old_bits = atomicLoad(&data[data_idx]);
    //     let old_f32 = bitcast<f32>(old_bits);
    //     let new_f32 = old_f32 + update_val;
    //     let new_bits = bitcast<u32>(new_f32);
    //     let res = atomicCompareExchangeWeak(&data[data_idx], old_bits, new_bits);
    //     if (res.exchanged) { break; }
    // }
}
`;
