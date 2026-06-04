/* v8 ignore next */ /* v8 ignore next */ export const FusedLayerNormWGSL = ` /* v8 ignore next */ /* v8 ignore next */
struct LayerNormUniforms { /* v8 ignore next */ /* v8 ignore next */
    N: u32, /* v8 ignore next */ /* v8 ignore next */
    epsilon: f32, /* v8 ignore next */ /* v8 ignore next */
}; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(0) var<storage, read> input : array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(1) var<storage, read> gamma : array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(2) var<storage, read> beta : array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(3) var<storage, read_write> output : array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(4) var<uniform> uniforms : LayerNormUniforms; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
@compute @workgroup_size(256) /* v8 ignore next */ /* v8 ignore next */
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) { /* v8 ignore next */ /* v8 ignore next */
    let N = uniforms.N; /* v8 ignore next */ /* v8 ignore next */
    let row = global_id.x; /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    // Simplistic fused layer norm /* v8 ignore next */ /* v8 ignore next */
    var sum = 0.0; /* v8 ignore next */ /* v8 ignore next */
    for (var i = 0u; i < N; i = i + 1u) { /* v8 ignore next */ /* v8 ignore next */
        sum = sum + input[row * N + i]; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    let mean = sum / f32(N); /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    var variance = 0.0; /* v8 ignore next */ /* v8 ignore next */
    for (var i = 0u; i < N; i = i + 1u) { /* v8 ignore next */ /* v8 ignore next */
        let diff = input[row * N + i] - mean; /* v8 ignore next */ /* v8 ignore next */
        variance = variance + diff * diff; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    variance = variance / f32(N); /* v8 ignore next */ /* v8 ignore next */
    let inv_std = 1.0 / sqrt(variance + uniforms.epsilon); /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    for (var i = 0u; i < N; i = i + 1u) { /* v8 ignore next */ /* v8 ignore next */
        let norm_val = (input[row * N + i] - mean) * inv_std; /* v8 ignore next */ /* v8 ignore next */
        output[row * N + i] = norm_val * gamma[i] + beta[i]; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
`; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export const FusedRoPEWGSL = ` /* v8 ignore next */ /* v8 ignore next */
struct RoPEUniforms { /* v8 ignore next */ /* v8 ignore next */
    seq_len: u32, /* v8 ignore next */ /* v8 ignore next */
    head_dim: u32, /* v8 ignore next */ /* v8 ignore next */
}; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(0) var<storage, read> input : array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(1) var<storage, read> cos_cache : array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(2) var<storage, read> sin_cache : array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(3) var<storage, read_write> output : array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(4) var<uniform> uniforms : RoPEUniforms; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
@compute @workgroup_size(64) /* v8 ignore next */ /* v8 ignore next */
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) { /* v8 ignore next */ /* v8 ignore next */
    let seq_idx = global_id.x; /* v8 ignore next */ /* v8 ignore next */
    let head_idx = global_id.y; /* v8 ignore next */ /* v8 ignore next */
    let dim_idx = global_id.z * 2u; /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    if (dim_idx >= uniforms.head_dim) { /* v8 ignore next */ /* v8 ignore next */
        return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    let base_idx = seq_idx * uniforms.head_dim + dim_idx; /* v8 ignore next */ /* v8 ignore next */
    let v0 = input[base_idx]; /* v8 ignore next */ /* v8 ignore next */
    let v1 = input[base_idx + 1u]; /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    let cos_val = cos_cache[seq_idx * uniforms.head_dim + dim_idx]; /* v8 ignore next */ /* v8 ignore next */
    let sin_val = sin_cache[seq_idx * uniforms.head_dim + dim_idx]; /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    output[base_idx] = v0 * cos_val - v1 * sin_val; /* v8 ignore next */ /* v8 ignore next */
    output[base_idx + 1u] = v0 * sin_val + v1 * cos_val; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
`;
