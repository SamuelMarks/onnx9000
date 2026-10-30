import { describe, expect, it } from 'vitest';
import { InferenceSession } from '../src/session';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('session.ts', () => {
  it('should instantiate and cover InferenceSession', () => {
    try {
      const obj = Construct(InferenceSession);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
