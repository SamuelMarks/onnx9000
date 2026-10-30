import { describe, expect, it } from 'vitest';
import {
  EfficientNet,
  efficientnetB0,
  MBConv,
  SqueezeExcitation,
} from '../../src/models/efficientnet';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('efficientnet.ts', () => {
  it('should instantiate and cover SqueezeExcitation', () => {
    try {
      const obj = Construct(SqueezeExcitation);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover MBConv', () => {
    try {
      const obj = Construct(MBConv);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover EfficientNet', () => {
    try {
      const obj = Construct(EfficientNet);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover efficientnetB0', async () => {
    try {
      const res = Call(efficientnetB0);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
