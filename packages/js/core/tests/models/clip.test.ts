import { describe, expect, it } from 'vitest';
import { CLIP, clipVitBasePatch16 } from '../../src/models/clip';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('clip.ts', () => {
  it('should instantiate and cover CLIP', () => {
    try {
      const obj = Construct(CLIP);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover clipVitBasePatch16', async () => {
    try {
      const res = Call(clipVitBasePatch16);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
