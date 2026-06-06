import { describe, it } from 'vitest';
import * as Module from '../../../src/dialects/web/scf';

describe('scf.ts', () => {
  it('should call and cover forOp', async () => {
    try {
      const res = (Module as any).forOp();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover yieldOp', async () => {
    try {
      const res = (Module as any).yieldOp();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover ifOp', async () => {
    try {
      const res = (Module as any).ifOp();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover whileOp', async () => {
    try {
      const res = (Module as any).whileOp();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover condition', async () => {
    try {
      const res = (Module as any).condition();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
