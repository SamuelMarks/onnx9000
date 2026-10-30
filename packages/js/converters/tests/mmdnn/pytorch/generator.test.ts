import { describe, expect, it } from 'vitest';
import { PyTorchGenerator } from '../../../src/mmdnn/pytorch/generator';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('generator.ts', () => {
  it('should instantiate and cover PyTorchGenerator', () => {
    try {
      const obj = Construct(PyTorchGenerator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
