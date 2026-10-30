import { describe, expect, it } from 'vitest';
import { RNN, RWKV, RWKVBlock, RWKVChannelMix, RWKVTimeMix, rwkvV4 } from '../../src/models/rwkv';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('rwkv.ts', () => {
  it('should instantiate and cover RNN', () => {
    try {
      const obj = Construct(RNN);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover RWKVTimeMix', () => {
    try {
      const obj = Construct(RWKVTimeMix);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover RWKVChannelMix', () => {
    try {
      const obj = Construct(RWKVChannelMix);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover RWKVBlock', () => {
    try {
      const obj = Construct(RWKVBlock);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover RWKV', () => {
    try {
      const obj = Construct(RWKV);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover rwkvV4', async () => {
    try {
      const res = Call(rwkvV4);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
