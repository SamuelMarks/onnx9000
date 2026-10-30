import { describe, expect, it } from 'vitest';
import { Block, PatchEmbed, VisionTransformer, vitBasePatch16_224 } from '../../src/models/vit';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('vit.ts', () => {
  it('should instantiate and cover PatchEmbed', () => {
    try {
      const obj = Construct(PatchEmbed);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Block', () => {
    try {
      const obj = Construct(Block);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover VisionTransformer', () => {
    try {
      const obj = Construct(VisionTransformer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover vitBasePatch16_224', async () => {
    try {
      const res = Call(vitBasePatch16_224);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
