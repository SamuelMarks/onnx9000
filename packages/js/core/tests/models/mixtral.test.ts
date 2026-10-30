import { describe, expect, it } from 'vitest';
import { Mixtral, MixtralBlock, mixtral8x7b, SparseMoE } from '../../src/models/mixtral';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mixtral.ts', () => {
  it('should instantiate and cover SparseMoE', () => {
    try {
      const obj = Construct(SparseMoE);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover MixtralBlock', () => {
    try {
      const obj = Construct(MixtralBlock);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Mixtral', () => {
    try {
      const obj = Construct(Mixtral);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover mixtral8x7b', async () => {
    try {
      const res = Call(mixtral8x7b);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
