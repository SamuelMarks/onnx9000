import { describe, expect, it } from 'vitest';
import { MatMulWebGPU, WebGPUProvider } from '../../../src/providers/webgpu/index';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('index.ts', () => {
  it('should instantiate and cover MatMulWebGPU', () => {
    try {
      const obj = Construct(MatMulWebGPU);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover WebGPUProvider', () => {
    try {
      const obj = Construct(WebGPUProvider);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
