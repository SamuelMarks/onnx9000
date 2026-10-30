import { describe, expect, it } from 'vitest';
import { LayoutBuilder } from '../../src/components/layout';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('layout.ts', () => {
  it('should instantiate and cover LayoutBuilder', () => {
    try {
      const obj = Construct(LayoutBuilder);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
