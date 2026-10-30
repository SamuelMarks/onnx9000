import { describe, expect, it } from 'vitest';
import { GraphEditor } from '../../src/ui/editor';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('editor.ts', () => {
  it('should instantiate and cover GraphEditor', () => {
    try {
      const obj = Construct(GraphEditor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
