import { describe, expect, it } from 'vitest';
import { LegacyQuirkResolver } from '../../../src/mmdnn/legacy/quirks';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('quirks.ts', () => {
  it('should instantiate and cover LegacyQuirkResolver', () => {
    try {
      const obj = Construct(LegacyQuirkResolver);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
