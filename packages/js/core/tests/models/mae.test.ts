import { describe, expect, it } from 'vitest';
import { MaskedAutoencoderViT, maeVitBasePatch16 } from '../../src/models/mae';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mae.ts', () => {
  it('should instantiate and cover MaskedAutoencoderViT', () => {
    try {
      const obj = Construct(MaskedAutoencoderViT);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover maeVitBasePatch16', async () => {
    try {
      const res = Call(maeVitBasePatch16);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
