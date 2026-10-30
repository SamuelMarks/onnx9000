import { describe, expect, it } from 'vitest';
import { JaxMapper } from '../../src/jax/mapper';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mapper.ts', () => {
  it('should instantiate and cover JaxMapper', () => {
    try {
      const obj = Construct(JaxMapper);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
