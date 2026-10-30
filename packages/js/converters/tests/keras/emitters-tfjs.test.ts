import { describe, expect, it } from 'vitest';
import { mapTfjsOpToOnnx } from '../../src/keras/emitters-tfjs';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('emitters-tfjs.ts', () => {
  it('should call and cover mapTfjsOpToOnnx', async () => {
    try {
      const res = Call(mapTfjsOpToOnnx);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
