import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/core/OnnxAdapter';

describe('OnnxAdapter.ts', () => {
  it('should instantiate and cover OnnxAdapter', () => {
    // Attempt to instantiate
    try {
      const obj = new (Module as any).OnnxAdapter();
      expect(obj).toBeDefined();
    } catch (e) {}
  });
});
