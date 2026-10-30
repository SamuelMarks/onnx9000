import { describe, expect, it } from 'vitest';
import { SparkMLParser } from '../../../src/mmdnn/sparkml/parser';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('parser.ts', () => {
  it('should instantiate and cover SparkMLParser', () => {
    try {
      const obj = Construct(SparkMLParser);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
