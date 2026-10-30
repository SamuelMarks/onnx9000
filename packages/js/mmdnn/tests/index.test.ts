import { describe, expect, it } from 'vitest';
import { Mmdnn } from '../src/index.js';

describe('Mmdnn', () => {
  it('should map legacy framework layers to ONNX operators', () => {
    const m = new Mmdnn();
    expect(m.run()).toBe('[mmdnn] processed');
    expect(m.mapLayerToOnnx({ framework: 'caffe', layerType: 'Convolution', name: 'conv1' })).toBe(
      'Conv',
    );
    expect(m.mapLayerToOnnx({ framework: 'ncnn', layerType: 'CustomNcnnOp', name: 'c1' })).toBe(
      'CustomNcnnOp',
    );
  });
});
