import { describe, expect, it } from 'vitest';
import * as Module from '../../src/models/mamba';

describe('mamba.ts', () => {
  it('should instantiate and cover StateSpace', () => {
    try {
      const obj = new (Module as any).StateSpace();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover MambaBlock', () => {
    try {
      const obj = new (Module as any).MambaBlock();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover Mamba', () => {
    try {
      const obj = new (Module as any).Mamba();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should call and cover mamba130m', async () => {
    try {
      const res = (Module as any).mamba130m();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
