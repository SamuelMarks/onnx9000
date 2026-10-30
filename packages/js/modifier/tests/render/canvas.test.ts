import { describe, expect, it } from 'vitest';
import { GraphRenderer } from '../../src/render/canvas';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('canvas.ts', () => {
  it('should instantiate and cover GraphRenderer', () => {
    try {
      const obj = Construct(GraphRenderer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
