import { describe, it, expect } from 'vitest';
import { Keras2OnnxConverter } from '../src/keras/index.js';
import { Node } from '@onnx9000/core';

describe('Keras Phase 9 - Quantization-Aware Training & Dynamic Quantization', () => {
  it('should map QAT quantize_wrapper layers to QuantizeLinear and DequantizeLinear', () => {
    const modelJson = JSON.stringify({
      format: 'layers-model',
      weightsManifest: [],
      modelTopology: {
        class_name: 'Sequential',
        config: {
          layers: [
            { class_name: 'InputLayer', config: { name: 'in1', batch_input_shape: [null, 10], dtype: 'float32' } },
            { class_name: 'Dense', config: { name: 'quantize_wrapper_dense1', units: 10 } },
          ],
        },
      },
    });

    const converter = new Keras2OnnxConverter(modelJson);
    converter.convert();
    const finalNodes = (converter as any)._test_finalNodes as Node[];
    
    // There should be a QuantizeLinear and DequantizeLinear node preceding the MatMul in Dense
    const qNode = finalNodes.find(n => n.opType === 'QuantizeLinear');
    const dqNode = finalNodes.find(n => n.opType === 'DequantizeLinear');
    const matMulNode = finalNodes.find(n => n.opType === 'MatMul');
    
    expect(qNode).toBeDefined();
    expect(dqNode).toBeDefined();
    
    expect(qNode?.inputs[0]).toBe('in1:0:0');
    expect(dqNode?.inputs[0]).toBe(qNode?.outputs[0]);
    expect(matMulNode?.inputs[0]).toBe(dqNode?.outputs[0]);
  });

  it('should map 4-bit packed weights and dynamic_quant to MatMulNBits and DynamicQuantizeLinear', () => {
    const modelJson = JSON.stringify({
      format: 'layers-model',
      weightsManifest: [],
      modelTopology: {
        class_name: 'Sequential',
        config: {
          layers: [
            { class_name: 'InputLayer', config: { name: 'in1', batch_input_shape: [null, 10], dtype: 'float32' } },
            { class_name: 'Dense', config: { name: 'packed_4bit_dense', units: 10 } },
            { class_name: 'Dense', config: { name: 'dynamic_quant_dense', units: 10 } },
          ],
        },
      },
    });

    const converter = new Keras2OnnxConverter(modelJson);
    converter.convert();
    const finalNodes = (converter as any)._test_finalNodes as Node[];

    const matMulNBits = finalNodes.find(n => n.opType === 'MatMulNBits');
    expect(matMulNBits).toBeDefined();

    const dqlNode = finalNodes.find(n => n.opType === 'DynamicQuantizeLinear');
    expect(dqlNode).toBeDefined();
    expect(dqlNode?.inputs[0]).toBe('packed_4bit_dense:0:0'); // output of previous layer

    const matMulInteger = finalNodes.find(n => n.opType === 'MatMulInteger');
    expect(matMulInteger).toBeDefined();
    expect(matMulInteger?.inputs[0]).toBe(dqlNode?.outputs[0]); // Receives dynamically quantized input
  });
});
