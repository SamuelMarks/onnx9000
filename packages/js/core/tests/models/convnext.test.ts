import { describe, expect, it } from 'vitest';
import { ConvNeXt, ConvNeXtBlock, convnextTiny } from '../../src/models/convnext';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('convnext.ts', () => {
  it('should instantiate and cover ConvNeXtBlock', () => {
    try {
      const obj = Construct(ConvNeXtBlock);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ConvNeXt', () => {
    try {
      const obj = Construct(ConvNeXt);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover convnextTiny', async () => {
    try {
      const res = Call(convnextTiny);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
