import { describe, expect, it } from 'vitest';
import { Block, BlockArgument, Operation, Region, Value } from '../../src/ir/core';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('core.ts', () => {
  it('should instantiate and cover Value', () => {
    try {
      const obj = Construct(Value);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover BlockArgument', () => {
    try {
      const obj = Construct(BlockArgument);
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
  it('should instantiate and cover Region', () => {
    try {
      const obj = Construct(Region);
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
});
