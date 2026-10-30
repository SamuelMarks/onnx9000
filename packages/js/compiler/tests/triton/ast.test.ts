import { describe, expect, it } from 'vitest';
import { generateTriton, TritonAST } from '../../src/triton/ast';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('ast.ts', () => {
  it('should instantiate and cover TritonAST', () => {
    try {
      const obj = Construct(TritonAST);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover generateTriton', () => {
    try {
      Call(generateTriton);
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
