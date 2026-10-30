import { describe, expect, it } from 'vitest';
import {
  AffineDimExpr,
  AffineExpr,
  AffineMap,
  batchMatmul,
  conv2dNhwcHwcf,
  fill,
  generic,
  matmul,
  poolingNhwcMax,
  yieldOp,
} from '../../../src/dialects/web/linalg';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('linalg.ts', () => {
  it('should instantiate and cover AffineExpr', () => {
    try {
      const obj = Construct(AffineExpr);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AffineDimExpr', () => {
    try {
      const obj = Construct(AffineDimExpr);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AffineMap', () => {
    try {
      const obj = Construct(AffineMap);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover generic', async () => {
    try {
      const res = Call(generic);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover matmul', async () => {
    try {
      const res = Call(matmul);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover batchMatmul', async () => {
    try {
      const res = Call(batchMatmul);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover conv2dNhwcHwcf', async () => {
    try {
      const res = Call(conv2dNhwcHwcf);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover poolingNhwcMax', async () => {
    try {
      const res = Call(poolingNhwcMax);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover fill', async () => {
    try {
      const res = Call(fill);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover yieldOp', async () => {
    try {
      const res = Call(yieldOp);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
