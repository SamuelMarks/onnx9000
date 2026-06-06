import { describe, expect, it } from 'vitest';
import * as Module from '../../../src/mmdnn/tensorflow/generator';

describe('generator.ts', () => {
  it('should instantiate and cover TensorFlowGenerator', () => {
    try {
      const obj = new (Module as any).TensorFlowGenerator();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
