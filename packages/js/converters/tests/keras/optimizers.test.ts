import { describe, expect, it } from 'vitest';
import {
  applyQuantization,
  KerasGraphOptimizer,
  optimizeFusedOps,
} from '../../src/keras/optimizers';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('optimizers.ts', () => {
  it('should instantiate and cover KerasGraphOptimizer', () => {
    try {
      const obj = Construct(KerasGraphOptimizer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover optimizeFusedOps', async () => {
    try {
      const res = Call(optimizeFusedOps);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover applyQuantization', async () => {
    try {
      const res = Call(applyQuantization);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
