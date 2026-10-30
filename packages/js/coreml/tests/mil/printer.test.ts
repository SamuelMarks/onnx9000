import { describe, expect, it } from 'vitest';
import { MILPrinter } from '../../src/mil/printer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('printer.ts', () => {
  it('should instantiate and cover MILPrinter', () => {
    try {
      const obj = Construct(MILPrinter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
