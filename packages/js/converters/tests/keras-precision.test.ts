import { describe, it, expect } from 'vitest';
import { Keras2OnnxConverter } from '../src/keras/index.js';
import { Node } from '@onnx9000/core';

describe('Keras Phase 9 - Data Types & Precision', () => {
  it('should preserve type mixed_float16 by injecting Cast nodes', () => {
    const modelJson = JSON.stringify({
      format: 'layers-model',
      weightsManifest: [],
      modelTopology: {
        class_name: 'Sequential',
        config: {
          layers: [
            {
              class_name: 'InputLayer',
              config: { name: 'in1', batch_input_shape: [null, 10], dtype: 'float32' },
            },
            { class_name: 'Dense', config: { name: 'dense1', units: 10, dtype: 'mixed_float16' } },
          ],
        },
      },
    });

    const converter = new Keras2OnnxConverter(modelJson);
    converter.convert();
    const rawNodes = (converter as any).rawNodes as any[];

    // There should be a Cast node preceding the MatMul in Dense
    const castNode = rawNodes.find((n) => n.opType === 'Cast');
    expect(castNode).toBeDefined();
    expect(castNode?.attributes.find((a: any) => a.name === 'to')?.i).toBe(10); // 10 = float16
    expect(castNode?.inputs[0]).toBe('in1:0:0');

    const matMulNode = rawNodes.find((n) => n.opType === 'MatMul');
    expect(matMulNode?.inputs).toContain(castNode?.outputs[0]);
  });

  it('should map QKeras QDense to QLinearMatMul', () => {
    const modelJson = JSON.stringify({
      format: 'layers-model',
      weightsManifest: [],
      modelTopology: {
        class_name: 'Sequential',
        config: {
          layers: [
            {
              class_name: 'InputLayer',
              config: { name: 'in1', batch_input_shape: [null, 10], dtype: 'float32' },
            },
            { class_name: 'QDense', config: { name: 'qdense1', units: 10 } },
          ],
        },
      },
    });

    const converter = new Keras2OnnxConverter(modelJson);
    converter.convert();
    const rawNodes = (converter as any).rawNodes as any[];

    const qNode = rawNodes.find((n) => n.opType === 'QLinearMatMul');
    expect(qNode).toBeDefined();
    expect(qNode?.inputs).toEqual([
      'in1:0:0',
      'in1:0:0_scale',
      'in1:0:0_zp',
      'qdense1_weights',
      'qdense1_weights_scale',
      'qdense1_weights_zp',
      'qdense1:0:0_scale',
      'qdense1:0:0_zp',
    ]);
  });

  it('should map QKeras QConv2D to QLinearConv', () => {
    const modelJson = JSON.stringify({
      format: 'layers-model',
      weightsManifest: [],
      modelTopology: {
        class_name: 'Sequential',
        config: {
          layers: [
            {
              class_name: 'InputLayer',
              config: { name: 'in1', batch_input_shape: [null, 10, 10, 3], dtype: 'float32' },
            },
            { class_name: 'QConv2D', config: { name: 'qconv1', filters: 16, kernel_size: [3, 3] } },
          ],
        },
      },
    });

    const converter = new Keras2OnnxConverter(modelJson);
    converter.convert();
    const rawNodes = (converter as any).rawNodes as any[];

    const qNode = rawNodes.find((n) => n.opType === 'QLinearConv');
    expect(qNode).toBeDefined();
    expect(qNode?.inputs[3]).toBe('qconv1_kernel');
    expect(qNode?.attributes.find((a: any) => a.name === 'kernel_shape')?.ints).toEqual([3, 3]);
  });
});
