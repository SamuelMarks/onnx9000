// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { PipelineValidator } from '../src/core/PipelineValidator.js';

describe('PipelineValidator', () => {
  it('should validate transitions', () => {
    expect(PipelineValidator.isValidTransition('keras', '.onnx')).toBe(true);
    expect(PipelineValidator.isValidTransition('keras', 'mlir')).toBe(false);
    expect(PipelineValidator.getValidTargets('mlir')).toContain('iree-compiler');
  });
});
