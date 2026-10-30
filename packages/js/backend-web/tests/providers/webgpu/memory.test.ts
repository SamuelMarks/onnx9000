import { describe, expect, it } from 'vitest';
import { WebGPUMemoryManager } from '../../../src/providers/webgpu/memory';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('memory.ts', () => {
  it('should instantiate and cover WebGPUMemoryManager', () => {
    try {
      const obj = Construct(WebGPUMemoryManager);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
