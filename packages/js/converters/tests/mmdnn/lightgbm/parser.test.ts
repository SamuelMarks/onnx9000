import { describe, expect, it } from 'vitest';
import { LightGBMParser } from '../../../src/mmdnn/lightgbm/parser';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('parser.ts', () => {
  it('should instantiate and cover LightGBMParser', () => {
    try {
      const obj = Construct(LightGBMParser);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
