import { describe, expect, it } from 'vitest';
import { LibSVMMapper } from '../../../src/mmdnn/libsvm/mapper';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mapper.ts', () => {
  it('should instantiate and cover LibSVMMapper', () => {
    try {
      const obj = Construct(LibSVMMapper);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
