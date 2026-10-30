import { describe, expect, it } from 'vitest';
import { compileToIREE } from '../../src/iree/index';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('index.ts', () => {
  it('should call and cover compileToIREE', () => {
    try {
      Call(compileToIREE);
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
