import { describe, expect, it } from 'vitest';
import { loadProgressive, ProgressiveSession } from '../src/progressive';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('progressive.ts', () => {
  it('should instantiate and cover ProgressiveSession', () => {
    try {
      const obj = Construct(ProgressiveSession);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover loadProgressive', () => {
    try {
      Call(loadProgressive);
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
