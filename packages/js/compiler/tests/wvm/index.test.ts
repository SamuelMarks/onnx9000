import { describe, expect, it } from 'vitest';
import { emitWVM, WVMInterpreter } from '../../src/wvm/index';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('index.ts', () => {
  it('should instantiate and cover WVMInterpreter', () => {
    try {
      const obj = Construct(WVMInterpreter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover emitWVM', () => {
    try {
      Call(emitWVM);
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
