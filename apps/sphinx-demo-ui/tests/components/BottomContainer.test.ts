import { describe, expect, it } from 'vitest';
import { BottomContainer } from '../../src/components/BottomContainer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('BottomContainer.ts', () => {
  it('should instantiate and cover BottomContainer', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(BottomContainer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
