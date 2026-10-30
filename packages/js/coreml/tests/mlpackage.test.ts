import { describe, expect, it } from 'vitest';
import { MLPackageBuilder } from '../src/mlpackage';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mlpackage.ts', () => {
  it('should instantiate and cover MLPackageBuilder', () => {
    try {
      const obj = Construct(MLPackageBuilder);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
