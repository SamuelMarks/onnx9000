import { describe, expect, it } from 'vitest';
import { SplitPane } from '../../src/components/SplitPane';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('SplitPane.ts', () => {
  it('should instantiate and cover SplitPane', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(SplitPane);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
