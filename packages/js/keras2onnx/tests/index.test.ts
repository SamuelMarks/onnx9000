import { describe, expect, it } from 'vitest';
import { Keras2ONNXConverter } from '../src/index.js';

describe('Keras2ONNXConverter', () => {
  it('should convert Keras model strings and map layer specs', () => {
    const conv = new Keras2ONNXConverter();
    expect(conv.convert('sequential_v1')).toBe('[ONNX-IR] from keras sequential_v1');
    expect(() => conv.convert('')).toThrow('Invalid model string');

    const mapped = conv.mapLayers([
      { class_name: 'Conv2D', name: 'conv1' },
      { class_name: 'ReLU', name: 'relu1' },
      { class_name: 'Dense', name: 'fc1' },
      { class_name: 'CustomLayer', name: 'custom1' },
    ]);
    expect(mapped).toEqual([
      { opType: 'Conv', name: 'conv1' },
      { opType: 'Relu', name: 'relu1' },
      { opType: 'Gemm', name: 'fc1' },
      { opType: 'CustomLayer', name: 'custom1' },
    ]);
  });
});
