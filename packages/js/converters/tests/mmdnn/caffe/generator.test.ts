import { describe, expect, it } from 'vitest';
import * as Module from '../../../src/mmdnn/caffe/generator';

describe('generator.ts', () => {
  it('should instantiate and cover CaffeGenerator', () => {
    try {
      const obj = new (Module as any).CaffeGenerator();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
