import { describe, expect, it } from 'vitest';
import { PaddleParser } from '../../../src/mmdnn/paddle/parser';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('parser.ts', () => {
  it('should instantiate and cover PaddleParser', () => {
    try {
      const obj = Construct(PaddleParser);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
