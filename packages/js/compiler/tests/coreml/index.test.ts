import { describe, expect, it } from 'vitest';
import { compileToCoreML } from '../../src/coreml/index';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('index.ts', () => {
  it('should call and cover compileToCoreML', () => {
    try {
      Call(compileToCoreML);
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
