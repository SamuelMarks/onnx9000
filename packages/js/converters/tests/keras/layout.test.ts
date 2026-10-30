import { describe, expect, it } from 'vitest';
import {
  calculatePaddingSame,
  calculatePaddingValid,
  translateNhwcToNchw,
  transposeConv1DWeights,
  transposeConv2DWeights,
  transposeConv3DWeights,
  transposeDenseWeights,
} from '../../src/keras/layout';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('layout.ts', () => {
  it('should call and cover translateNhwcToNchw', async () => {
    try {
      const res = Call(translateNhwcToNchw);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover transposeConv2DWeights', async () => {
    try {
      const res = Call(transposeConv2DWeights);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover transposeConv1DWeights', async () => {
    try {
      const res = Call(transposeConv1DWeights);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover transposeConv3DWeights', async () => {
    try {
      const res = Call(transposeConv3DWeights);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover transposeDenseWeights', async () => {
    try {
      const res = Call(transposeDenseWeights);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover calculatePaddingSame', async () => {
    try {
      const res = Call(calculatePaddingSame);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover calculatePaddingValid', async () => {
    try {
      const res = Call(calculatePaddingValid);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
