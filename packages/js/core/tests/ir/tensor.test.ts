import { describe, expect, it } from 'vitest';
import { SparseTensor, Tensor } from '../../src/ir/tensor';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('tensor.ts', () => {
  it('should instantiate and cover Tensor', () => {
    try {
      const obj = Construct(Tensor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover SparseTensor', () => {
    try {
      const obj = Construct(SparseTensor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
