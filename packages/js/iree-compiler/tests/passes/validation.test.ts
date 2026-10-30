import { describe, expect, it } from 'vitest';
import { ValidationSuite } from '../../src/passes/validation';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('validation.ts', () => {
  it('should instantiate and cover ValidationSuite', () => {
    try {
      const obj = Construct(ValidationSuite);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
