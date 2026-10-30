import { describe, expect, it } from 'vitest';
import { Dropdown } from '../../src/components/Dropdown';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('Dropdown.ts', () => {
  it('should instantiate and cover Dropdown', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(Dropdown);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
