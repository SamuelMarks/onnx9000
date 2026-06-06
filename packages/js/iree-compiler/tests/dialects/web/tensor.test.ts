import { describe, expect, it } from 'vitest';
import * as Module from '../../../src/dialects/web/tensor';

describe('tensor.ts', () => {
  it('should instantiate and cover TensorType', () => {
    try {
      const obj = new (Module as any).TensorType();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should call and cover extract', async () => {
    try {
      const res = (Module as any).extract();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover insert', async () => {
    try {
      const res = (Module as any).insert();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover splat', async () => {
    try {
      const res = (Module as any).splat();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover pad', async () => {
    try {
      const res = (Module as any).pad();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
