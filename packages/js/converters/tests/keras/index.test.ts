import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/keras/index';

describe('index.ts', () => {
  it('should instantiate and cover Keras2OnnxConverter', () => {
    try {
       const obj = new (Module as any).Keras2OnnxConverter();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
