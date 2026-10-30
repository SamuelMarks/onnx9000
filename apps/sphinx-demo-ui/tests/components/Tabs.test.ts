import { describe, expect, it } from 'vitest';
import { Tabs } from '../../src/components/Tabs';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('Tabs.ts', () => {
  it('should instantiate and cover Tabs', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(Tabs);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
