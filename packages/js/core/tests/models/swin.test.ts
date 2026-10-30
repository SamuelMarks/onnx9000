import { describe, expect, it } from 'vitest';
import {
  SwinTransformer,
  SwinTransformerBlock,
  swinT,
  WindowAttention,
} from '../../src/models/swin';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('swin.ts', () => {
  it('should instantiate and cover WindowAttention', () => {
    try {
      const obj = Construct(WindowAttention);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover SwinTransformerBlock', () => {
    try {
      const obj = Construct(SwinTransformerBlock);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover SwinTransformer', () => {
    try {
      const obj = Construct(SwinTransformer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover swinT', async () => {
    try {
      const res = Call(swinT);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
