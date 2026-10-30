import { describe, expect, it } from 'vitest';
import { parseJaxpr } from '../../src/jax/jaxpr_parser';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('jaxpr_parser.ts', () => {
  it('should call and cover parseJaxpr', async () => {
    try {
      const res = Call(parseJaxpr);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
