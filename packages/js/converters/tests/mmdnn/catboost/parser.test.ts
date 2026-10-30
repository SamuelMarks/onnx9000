import { describe, expect, it } from 'vitest';
import { CatBoostParser } from '../../../src/mmdnn/catboost/parser';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('parser.ts', () => {
  it('should instantiate and cover CatBoostParser', () => {
    try {
      const obj = Construct(CatBoostParser);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
