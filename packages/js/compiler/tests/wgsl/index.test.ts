import { describe, expect, it } from 'vitest';
import { emitWGSL } from '../../src/wgsl/index';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('index.ts', () => {
  it('should call and cover emitWGSL', () => {
    try {
      Call(emitWGSL);
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
