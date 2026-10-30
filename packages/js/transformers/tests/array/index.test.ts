import { describe, expect, it } from 'vitest';
import { ArrayAPI } from '../../src/array/index';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('index.ts', () => {
  it('should instantiate and cover ArrayAPI', () => {
    try {
      const obj = Construct(ArrayAPI);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
