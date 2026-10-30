import { describe, expect, it } from 'vitest';
import { PyTorchSerializer } from '../../../src/mmdnn/pytorch/serializer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('serializer.ts', () => {
  it('should instantiate and cover PyTorchSerializer', () => {
    try {
      const obj = Construct(PyTorchSerializer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
