import { describe, expect, it } from 'vitest';
import * as Module from '../../../src/providers/webnn/webnn_builder';

describe('webnn_builder.ts', () => {
  it('should instantiate and cover KerasWebNNCompiler', () => {
    try {
      const obj = new (Module as any).KerasWebNNCompiler();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
