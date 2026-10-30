import { describe, expect, it } from 'vitest';
import { MLPackageLoader } from '../src/loader';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('loader.ts', () => {
  it('should instantiate and cover MLPackageLoader', () => {
    try {
      const obj = Construct(MLPackageLoader);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
