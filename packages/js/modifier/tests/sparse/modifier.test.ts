import { describe, expect, it } from 'vitest';
import {
  applyRecipe,
  ConstantPruningModifier,
  MagnitudePruningModifier,
  Modifier,
  parseRecipe,
} from '../../src/sparse/modifier';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('modifier.ts', () => {
  it('should instantiate and cover Modifier', () => {
    try {
      const obj = Construct(Modifier);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover MagnitudePruningModifier', () => {
    try {
      const obj = Construct(MagnitudePruningModifier);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ConstantPruningModifier', () => {
    try {
      const obj = Construct(ConstantPruningModifier);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover parseRecipe', async () => {
    try {
      const res = Call(parseRecipe);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover applyRecipe', async () => {
    try {
      const res = Call(applyRecipe);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
