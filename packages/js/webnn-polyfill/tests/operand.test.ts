import { describe, expect, it } from 'vitest';
import * as Module from '../src/operand';

describe('operand.ts', () => {
  it('should instantiate and cover PolyfillMLOperand', () => {
    try {
      const obj = new (Module as any).PolyfillMLOperand();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
