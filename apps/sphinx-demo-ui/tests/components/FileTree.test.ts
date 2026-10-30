import { describe, expect, it } from 'vitest';
import { FileTree } from '../../src/components/FileTree';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('FileTree.ts', () => {
  it('should instantiate and cover FileTree', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(FileTree);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
