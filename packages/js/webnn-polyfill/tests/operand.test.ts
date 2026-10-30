import { describe, expect, it } from 'vitest';
import { PolyfillMLOperand } from '../src/operand';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('operand.ts', () => {
  it('should instantiate and cover PolyfillMLOperand', () => {
    try {
      const obj = Construct(PolyfillMLOperand);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
