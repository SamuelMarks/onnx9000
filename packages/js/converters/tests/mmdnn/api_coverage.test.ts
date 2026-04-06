import { describe, it, expect } from 'vitest';
import {
  convertToPyTorch,
  convertToTensorFlow,
  convertToCaffe,
  convertToMXNet,
  convertToCNTK,
  convertToCoreML,
  convertToPaddle,
  convertToKeras,
  convertToOnnxScript,
} from '../../src/mmdnn/api.js';

describe('MMDNN API Coverage', () => {
  const dummyFile = new File([''], 'model.onnx', { type: 'application/octet-stream' });
  const files = [dummyFile];

  it('should call convertToPyTorch', async () => {
    const result = await convertToPyTorch(files);
    expect(result).toBeDefined();
    expect(result).toContain('import torch');
  });

  it('should call convertToTensorFlow', async () => {
    const result = await convertToTensorFlow(files);
    expect(result).toBeDefined();
    expect(result).toContain('import tensorflow');
  });

  it('should call convertToCaffe', async () => {
    const result = await convertToCaffe(files);
    expect(result).toBeDefined();
    expect(result).toContain('name: "Model"');
  });

  it('should call convertToMXNet', async () => {
    const result = await convertToMXNet(files);
    expect(result).toBeDefined();
    expect(result).toContain('import mxnet');
  });

  it('should call convertToCNTK', async () => {
    const result = await convertToCNTK(files);
    expect(result).toBeDefined();
    expect(result).toContain('import cntk');
  });

  it('should call convertToCoreML', async () => {
    const result = (await convertToCoreML(files)) as Object;
    expect(result).toBeDefined();
    expect(result.content).toContain('Exported coreml');
  });

  it('should call convertToPaddle', async () => {
    const result = (await convertToPaddle(files)) as Object;
    expect(result).toBeDefined();
    expect(result.content).toContain('Exported paddle');
  });

  it('should call convertToKeras', async () => {
    const result = await convertToKeras(files);
    expect(result).toBeDefined();
    expect(result).toContain('class Model_Generated(keras.Model):');
  });

  it('should call convertToOnnxScript', async () => {
    const result = await convertToOnnxScript(files);
    expect(result).toBeDefined();
    expect(result).toContain('from onnxscript import');
  });
});
