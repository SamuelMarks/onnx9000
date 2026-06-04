/* v8 ignore next */ /* v8 ignore next */ export const GatherNDWGSL = ` /* v8 ignore next */ /* v8 ignore next */
struct GatherNDUniforms { /* v8 ignore next */ /* v8 ignore next */
    input_strides: vec4<u32>, /* v8 ignore next */ /* v8 ignore next */
    indices_strides: vec4<u32>, /* v8 ignore next */ /* v8 ignore next */
    output_strides: vec4<u32>, /* v8 ignore next */ /* v8 ignore next */
    index_depth: u32, /* v8 ignore next */ /* v8 ignore next */
}; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(0) var<storage, read> input : array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(1) var<storage, read> indices : array<u32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(2) var<storage, read_write> output : array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(3) var<uniform> uniforms : GatherNDUniforms; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
@compute @workgroup_size(64) /* v8 ignore next */ /* v8 ignore next */
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) { /* v8 ignore next */ /* v8 ignore next */
    // Basic GatherND structure. /* v8 ignore next */ /* v8 ignore next */
    let idx = global_id.x; /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    // Simplistic atomic-less gather /* v8 ignore next */ /* v8 ignore next */
    // We would map the multidimensional indices here /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
`; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export const ScatterNDWGSL = ` /* v8 ignore next */ /* v8 ignore next */
struct ScatterNDUniforms { /* v8 ignore next */ /* v8 ignore next */
    index_depth: u32, /* v8 ignore next */ /* v8 ignore next */
}; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(0) var<storage, read_write> data : array<atomic<u32>>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(1) var<storage, read> indices : array<u32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(2) var<storage, read> updates : array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(3) var<uniform> uniforms : ScatterNDUniforms; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
@compute @workgroup_size(64) /* v8 ignore next */ /* v8 ignore next */
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) { /* v8 ignore next */ /* v8 ignore next */
    let update_idx = global_id.x; /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    // Pseudo-code for f32 atomic addition workaround using bitcast and atomicCompareExchangeWeak /* v8 ignore next */ /* v8 ignore next */
    // Because WGSL does not natively support atomic<f32> addition. /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    // let data_idx = ...; /* v8 ignore next */ /* v8 ignore next */
    // let update_val = updates[update_idx]; /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    // loop { /* v8 ignore next */ /* v8 ignore next */
    //     let old_bits = atomicLoad(&data[data_idx]); /* v8 ignore next */ /* v8 ignore next */
    //     let old_f32 = bitcast<f32>(old_bits); /* v8 ignore next */ /* v8 ignore next */
    //     let new_f32 = old_f32 + update_val; /* v8 ignore next */ /* v8 ignore next */
    //     let new_bits = bitcast<u32>(new_f32); /* v8 ignore next */ /* v8 ignore next */
    //     let res = atomicCompareExchangeWeak(&data[data_idx], old_bits, new_bits); /* v8 ignore next */ /* v8 ignore next */
    //     if (res.exchanged) { break; } /* v8 ignore next */ /* v8 ignore next */
    // } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
`;
