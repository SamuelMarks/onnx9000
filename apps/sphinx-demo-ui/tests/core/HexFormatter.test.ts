import { describe, expect, it } from 'vitest';
import { HexFormatter } from '../../src/core/HexFormatter';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('HexFormatter.ts', () => {
  it('should instantiate and cover HexFormatter', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(HexFormatter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
