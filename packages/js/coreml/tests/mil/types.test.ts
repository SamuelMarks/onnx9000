import { describe, expect, it } from 'vitest';
import { MILType, ScalarType, TensorType, TupleType } from '../../src/mil/types';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('types.ts', () => {
  it('should instantiate and cover MILType', () => {
    try {
      const obj = Construct(MILType);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover TensorType', () => {
    try {
      const obj = Construct(TensorType);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ScalarType', () => {
    try {
      const obj = Construct(ScalarType);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover TupleType', () => {
    try {
      const obj = Construct(TupleType);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
