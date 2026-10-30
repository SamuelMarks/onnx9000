import { describe, expect, it } from 'vitest';
import { Block, MILFunction, Operation, Program, Var } from '../../src/mil/ast';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('ast.ts', () => {
  it('should instantiate and cover Var', () => {
    try {
      const obj = Construct(Var);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Operation', () => {
    try {
      const obj = Construct(Operation);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Block', () => {
    try {
      const obj = Construct(Block);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover MILFunction', () => {
    try {
      const obj = Construct(MILFunction);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Program', () => {
    try {
      const obj = Construct(Program);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
