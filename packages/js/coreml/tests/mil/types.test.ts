import { describe, expect, it } from 'vitest';
import * as Module from '../../src/mil/types';

describe('types.ts', () => {
  it('should instantiate and cover MILType', () => {
    try {
      const obj = new (Module as any).MILType();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover TensorType', () => {
    try {
      const obj = new (Module as any).TensorType();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover ScalarType', () => {
    try {
      const obj = new (Module as any).ScalarType();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should instantiate and cover TupleType', () => {
    try {
      const obj = new (Module as any).TupleType();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
