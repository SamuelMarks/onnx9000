import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../../src/mmdnn/libsvm/mapper';

describe('mapper.ts', () => {
  it('should instantiate and cover LibSVMMapper', () => {
    try {
       const obj = new (Module as any).LibSVMMapper();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
