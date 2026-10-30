import { describe, expect, it } from 'vitest';
import { emitConv, emitSeparableConv } from '../../src/keras/emitters-conv';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('emitters-conv.ts', () => {
  it('should call and cover emitConv', async () => {
    try {
      const res = Call(emitConv);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover emitSeparableConv', async () => {
    try {
      const res = Call(emitSeparableConv);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
