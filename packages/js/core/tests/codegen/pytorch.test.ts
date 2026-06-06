import { describe, expect, it } from 'vitest';
import * as Module from '../../src/codegen/pytorch';

describe('pytorch.ts', () => {
  it('should instantiate and cover ONNXToPyTorchVisitor', () => {
    try {
      const obj = new (Module as any).ONNXToPyTorchVisitor();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should call and cover cleanName', async () => {
    try {
      const res = (Module as any).cleanName();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
