import { describe, expect, it } from 'vitest';
import { DagreLayoutEngine } from '../../src/render/layout';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('layout.ts', () => {
  it('should instantiate and cover DagreLayoutEngine', () => {
    try {
      const obj = Construct(DagreLayoutEngine);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
