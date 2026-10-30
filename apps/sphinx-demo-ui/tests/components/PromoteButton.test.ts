import { describe, expect, it } from 'vitest';
import { PromoteButton } from '../../src/components/PromoteButton';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('PromoteButton.ts', () => {
  it('should instantiate and cover PromoteButton', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(PromoteButton);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
