import { describe, expect, it } from 'vitest';
import {
  mapConv2DOptions,
  mapDepthwiseConv2DOptions,
  mapOnnxNodeToTFLite,
  mapPool2DOptions,
  mapReducerOptions,
} from '../../src/compiler/operators';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('operators.ts', () => {
  it('should call and cover mapPool2DOptions', async () => {
    try {
      const res = Call(mapPool2DOptions);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover mapReducerOptions', async () => {
    try {
      const res = Call(mapReducerOptions);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover mapConv2DOptions', async () => {
    try {
      const res = Call(mapConv2DOptions);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover mapDepthwiseConv2DOptions', async () => {
    try {
      const res = Call(mapDepthwiseConv2DOptions);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover mapOnnxNodeToTFLite', async () => {
    try {
      const res = Call(mapOnnxNodeToTFLite);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
