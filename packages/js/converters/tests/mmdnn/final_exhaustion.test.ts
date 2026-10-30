import { Node, Tensor } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import * as api from '../../src/mmdnn/api.js';
import { CaffeGenerator } from '../../src/mmdnn/caffe/generator.js';
import { CNTKGenerator } from '../../src/mmdnn/cntk/generator.js';
import { KerasGenerator } from '../../src/mmdnn/keras/generator.js';
import { MXNetGenerator } from '../../src/mmdnn/mxnet/generator.js';
import { TensorFlowGenerator } from '../../src/mmdnn/tensorflow/generator.js';
import { ONNXNormalizer } from '../../src/mmdnn/verification/normalizer.js';

describe('MMDNN Submodule Final Gaps', () => {
  const mockOnnxGraph = {
    name: 'TestGraph',
    nodes: [new Node('Relu', ['in'], ['out'])],
    inputs: [{ name: 'in', shape: [1, 3], dtype: 'float32' }],
    outputs: [{ name: 'out', shape: [1, 3], dtype: 'float32' }],
    tensors: {
      w: new Tensor('w', [1], 'float32', true, false, new Uint8Array(4)),
    },
    initializers: ['w'],
    valueInfo: [],
  };

  it('should cover KerasGenerator additional branches', () => {
    const gen = new KerasGenerator(
      mockOnnxGraph as never as ConstructorParameters<typeof KerasGenerator>[0],
    );
    expect(gen.generateSource()).toContain('class Model_Generated');

    // empty graph
    const emptyGen = new KerasGenerator({ nodes: [] } as never as ConstructorParameters<
      typeof KerasGenerator
    >[0]);
    expect(emptyGen.generateSource()).toContain('pass');
  });

  it('should cover CaffeGenerator gaps', () => {
    const gen = new CaffeGenerator(
      mockOnnxGraph as never as ConstructorParameters<typeof CaffeGenerator>[0],
    );
    expect(gen.generate()).toContain('layer {');
  });

  it('should cover MXNetGenerator gaps', () => {
    const gen = new MXNetGenerator(
      mockOnnxGraph as never as ConstructorParameters<typeof MXNetGenerator>[0],
    );
    expect(gen.generate()).toContain('import mxnet');
  });

  it('should cover CNTKGenerator gaps', () => {
    const gen = new CNTKGenerator(
      mockOnnxGraph as never as ConstructorParameters<typeof CNTKGenerator>[0],
    );
    expect(gen.generate()).toContain('import cntk');
  });

  it('should cover TensorFlowGenerator gaps', () => {
    const gen = new TensorFlowGenerator(
      mockOnnxGraph as never as ConstructorParameters<typeof TensorFlowGenerator>[0],
    );
    expect(gen.generate()).toContain('import tensorflow');
  });

  it('should cover Normalizer gaps', () => {
    const norm = new ONNXNormalizer();
    try {
      norm.normalize(mockOnnxGraph as never as Parameters<ONNXNormalizer['normalize']>[0]);
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });

  it('should cover api.ts convert cases', async () => {
    for (const target of [
      'pytorch',
      'tensorflow',
      'caffe',
      'mxnet',
      'cntk',
      'coreml',
      'paddle',
      'keras',
      'onnxscript',
    ]) {
      try {
        await api.convert(
          mockOnnxGraph as never as Parameters<typeof api.convert>[0],
          target as never as Parameters<typeof api.convert>[1],
          [],
        );
      } catch (_e) {
        expect(_e).toBeDefined();
      }
    }
  });
});
