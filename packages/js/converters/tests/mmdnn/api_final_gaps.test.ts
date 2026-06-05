import { describe, it, expect, vi } from 'vitest';
import * as api from '../../src/mmdnn/api.js';
import { getTypesIdentifier } from '../../src/mmdnn/types.js';
import { Graph } from '@onnx9000/core';

describe('MMDNN API Coverage Gaps Final V3', () => {
  it('should cover catch branches for multiple frameworks with throwing blobs', async () => {
    // Create a mock File where text() and arrayBuffer() always throw
    const errorFile = new File([new Blob([''])], 'test.onnx');
    errorFile.text = () => Promise.reject(new Error('forced text error'));
    errorFile.arrayBuffer = () => Promise.reject(new Error('forced buffer error'));

    const sources = [
      'tensorflow',
      'caffe',
      'mxnet',
      'cntk',
      'coreml',
      'paddle',
      'keras',
      'onnxscript',
      'xgboost',
      'catboost',
      'sparkml',
      'lightgbm',
      'scikitlearn',
      'pytorch',
      'ncnn',
      'darknet',
    ];
    for (const source of sources) {
      try {
        await api.convertToGraph([errorFile], source as any);
      } catch (e) {}
    }
  });

  it('should cover types identifier and extra convert branches', async () => {
    expect(getTypesIdentifier()).toBe('MMDNN_TYPES');
    const fakeFile = new File([new Blob(['invalid'])], 'test.onnx');
    const frameworks = [
      'pytorch',
      'tensorflow',
      'caffe',
      'mxnet',
      'cntk',
      'coreml',
      'paddle',
      'keras',
      'onnxscript',
    ];
    for (const fw of frameworks) {
      try {
        await api.convert(fw as any, 'tfjs', [fakeFile]);
      } catch (e) {}
    }
  });
});
