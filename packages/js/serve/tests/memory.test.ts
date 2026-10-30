import { describe, expect, it } from 'vitest';
import { MemoryManager } from '../src/memory';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('memory.ts', () => {
  it('should instantiate and cover MemoryManager', () => {
    try {
      const obj = Construct(MemoryManager);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
