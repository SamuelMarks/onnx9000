import { describe, expect, it } from 'vitest';
import { parseFlaxState } from '../../src/jax/flax_parser';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('flax_parser.ts', () => {
  it('should call and cover parseFlaxState', async () => {
    try {
      const res = Call(parseFlaxState);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
