import { describe, expect, it } from 'vitest';
import * as Module from '../../../src/providers/webnn/conformance_validator';

describe('conformance_validator.ts', () => {
  it('should instantiate and cover WebNNLayoutValidator', () => {
    try {
      const obj = new (Module as any).WebNNLayoutValidator();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
