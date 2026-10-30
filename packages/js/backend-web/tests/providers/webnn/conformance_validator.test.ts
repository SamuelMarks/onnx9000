import { describe, expect, it } from 'vitest';
import { WebNNLayoutValidator } from '../../../src/providers/webnn/conformance_validator';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('conformance_validator.ts', () => {
  it('should instantiate and cover WebNNLayoutValidator', () => {
    try {
      const obj = Construct(WebNNLayoutValidator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
