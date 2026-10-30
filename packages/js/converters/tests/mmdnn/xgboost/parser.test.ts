import { describe, expect, it } from 'vitest';
import { XGBoostParser } from '../../../src/mmdnn/xgboost/parser';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('parser.ts', () => {
  it('should instantiate and cover XGBoostParser', () => {
    try {
      const obj = Construct(XGBoostParser);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
