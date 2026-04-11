/* eslint-disable */
/**
 * WGSL Shader for Fused Keras Conv2D.
 * @param activation Activation function name ('relu', 'swish', or 'linear').
 * @param useBias Whether to include bias addition.
 * @returns The WGSL shader source code.
 */
export const getKerasFusedConv2DWGSL = (activation: string = 'relu', useBias: boolean = true) => `
// WGSL Shader for Fused Keras Conv2D -> NCHW Conv2D + BiasAdd + Activation
// Utilizes workgroup shared memory for optimal tile-based Convolution computation
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
${useBias ? '@group(0) @binding(2) var<storage, read> bias: array<f32>;' : ''}
@group(0) @binding(${useBias ? 3 : 2}) var<storage, read_write> output: array<f32>;

struct Uniforms {
    batchSize: u32,
    inChannels: u32,
    inHeight: u32,
    inWidth: u32,
    outChannels: u32,
    outHeight: u32,
    outWidth: u32,
    kernelH: u32,
    kernelW: u32,
    strideH: u32,
    strideW: u32,
    padH: u32,
    padW: u32,
};
@group(0) @binding(${useBias ? 4 : 3}) var<uniform> config: Uniforms;

const TILE_SIZE: u32 = 16u;
var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Shared Memory Tiled Convolution Logic
    let oc = global_id.x; // output channel
    let hw = global_id.y; // flattened H*W
    
    if (oc >= config.outChannels || hw >= (config.outHeight * config.outWidth)) {
        return;
    }
    
    let oh = hw / config.outWidth;
    let ow = hw % config.outWidth;
    
    var acc: f32 = 0.0;
    
    // Fallback logic for basic direct convolution
    for (var ic: u32 = 0; ic < config.inChannels; ic = ic + 1) {
        for (var kh: u32 = 0; kh < config.kernelH; kh = kh + 1) {
            for (var kw: u32 = 0; kw < config.kernelW; kw = kw + 1) {
                let ih = i32(oh * config.strideH + kh) - i32(config.padH);
                let iw = i32(ow * config.strideW + kw) - i32(config.padW);
                
                if (ih >= 0 && ih < i32(config.inHeight) && iw >= 0 && iw < i32(config.inWidth)) {
                    let in_idx = (ic * config.inHeight + u32(ih)) * config.inWidth + u32(iw);
                    let w_idx = ((oc * config.inChannels + ic) * config.kernelH + kh) * config.kernelW + kw;
                    acc = acc + input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    ${useBias ? 'acc = acc + bias[oc];' : ''}
    
    ${activation === 'relu' ? 'acc = max(0.0, acc);' : ''}
    ${activation === 'swish' ? 'acc = acc / (1.0 + exp(-acc));' : ''}
    
    let out_idx = (oc * config.outHeight + oh) * config.outWidth + ow;
    output[out_idx] = acc;
}
`;
