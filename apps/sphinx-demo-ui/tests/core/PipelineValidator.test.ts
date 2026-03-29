/* eslint-disable */
// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { PipelineValidator } from '../../src/core/PipelineValidator';

describe('PipelineValidator', () => {
  it('should validate correct transitions', () => {
    expect(PipelineValidator.isValidTransition('.onnx', 'olive')).toBe(true);
    expect(PipelineValidator.isValidTransition('.ONNX', 'OLIVE')).toBe(true); // Case insensitive
    expect(PipelineValidator.isValidTransition('.onnx', 'onnx-mlir')).toBe(true);
    expect(PipelineValidator.isValidTransition('.onnx', 'onnx2c')).toBe(true);
    expect(PipelineValidator.isValidTransition('.onnx', 'ort-web')).toBe(true);

    expect(PipelineValidator.isValidTransition('mlir', 'iree-compiler')).toBe(true);
    expect(PipelineValidator.isValidTransition('cpp', 'emscripten')).toBe(true);

    expect(PipelineValidator.isValidTransition('keras', '.onnx')).toBe(true);
    expect(PipelineValidator.isValidTransition('tensorflow', '.onnx')).toBe(true);
  });

  it('should reject invalid transitions', () => {
    expect(PipelineValidator.isValidTransition('keras', 'iree-compiler')).toBe(false);
    expect(PipelineValidator.isValidTransition('mlir', 'olive')).toBe(false);
    expect(PipelineValidator.isValidTransition('.onnx', 'emscripten')).toBe(false);
  });

  it('should return empty targets for unknown sources', () => {
    expect(PipelineValidator.getValidTargets('unknown_source')).toEqual([]);
    expect(PipelineValidator.isValidTransition('unknown_source', '.onnx')).toBe(false);
  });

  it('should return valid targets list', () => {
    expect(PipelineValidator.getValidTargets('.onnx')).toContain('olive');
    expect(PipelineValidator.getValidTargets('.onnx')).toContain('onnx-mlir');
  });
});
