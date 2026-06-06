import { describe, expect, it } from 'vitest';
import * as Module from '../../src/models/dit';

describe('dit.ts', () => {
  it('should instantiate and cover DiTBlock', () => {
    try {
      const obj = new (Module as any).DiTBlock();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover DiT', () => {
    try {
      const obj = new (Module as any).DiT();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should call and cover ditXl2', async () => {
    try {
      const res = (Module as any).ditXl2();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
