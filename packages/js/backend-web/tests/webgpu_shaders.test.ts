import { describe, it, expect } from 'vitest';
import { getKerasFusedConv2DWGSL } from '../src/providers/webgpu/shaders/keras_fused_conv2d.js';
import { getKerasLayerNormWGSL } from '../src/providers/webgpu/shaders/keras_layer_norm.js';

describe('WebGPU Shaders', () => {
  it('should generate Keras Fused Conv2D shader', () => {
    const shader = getKerasFusedConv2DWGSL('relu', true);
    expect(shader).toContain('Conv2D');
    expect(shader).toContain('max(0.0, acc)');

    const swishShader = getKerasFusedConv2DWGSL('swish', false);
    expect(swishShader).toContain('acc / (1.0 + exp(-acc))');
    expect(swishShader).not.toContain('bias');
  });

  it('should generate Keras LayerNorm shader', () => {
    const shader = getKerasLayerNormWGSL();
    expect(shader).toContain('inverseSqrt(variance + config.epsilon)');
  });
});
