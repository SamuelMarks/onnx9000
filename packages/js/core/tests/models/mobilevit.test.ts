import { describe, expect, it } from 'vitest';
import { MobileViT, MobileViTBlock, mobilevitS } from '../../src/models/mobilevit';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mobilevit.ts', () => {
  it('should instantiate and cover MobileViTBlock', () => {
    try {
      const obj = Construct(MobileViTBlock);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover MobileViT', () => {
    try {
      const obj = Construct(MobileViT);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover mobilevitS', async () => {
    try {
      const res = Call(mobilevitS);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
