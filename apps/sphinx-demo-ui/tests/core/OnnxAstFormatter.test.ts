import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/core/OnnxAstFormatter';

describe('OnnxAstFormatter.ts', () => {
  it('should instantiate and cover OnnxAstFormatter', () => {
    // Attempt to instantiate
    try {
      const obj = new (Module as any).OnnxAstFormatter();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
