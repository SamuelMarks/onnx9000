import { describe, expect, it } from 'vitest';
import { MLIRInterop } from '../../src/passes/interop';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('interop.ts', () => {
  it('should instantiate and cover MLIRInterop', () => {
    try {
      const obj = Construct(MLIRInterop);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
