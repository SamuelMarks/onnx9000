// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { PipelineValidator } from '../src/core/PipelineValidator.js';

describe('PipelineValidator', () => {
  it('should validate transitions', () => {
    expect(PipelineValidator.isValidTransition('keras', '.onnx')).toBe(true);
    expect(PipelineValidator.isValidTransition('keras', 'mlir')).toBe(false);
    expect(PipelineValidator.getValidTargets('mlir')).toContain('iree-compiler');
  });
});
