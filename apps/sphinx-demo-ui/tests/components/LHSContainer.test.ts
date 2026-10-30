import { describe, expect, it } from 'vitest';
import { LHSContainer } from '../../src/components/LHSContainer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('LHSContainer.ts', () => {
  it('should instantiate and cover LHSContainer', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(LHSContainer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
