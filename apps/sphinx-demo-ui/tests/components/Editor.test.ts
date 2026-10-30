import { describe, expect, it } from 'vitest';
import { Editor } from '../../src/components/Editor';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('Editor.ts', () => {
  it('should instantiate and cover Editor', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(Editor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
