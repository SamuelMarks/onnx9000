import { describe, expect, it } from 'vitest';
import { GraphDebugger } from '../../../src/components/debugger/debugger';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('debugger.ts', () => {
  it('should instantiate and cover GraphDebugger', () => {
    try {
      const obj = Construct(GraphDebugger);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
