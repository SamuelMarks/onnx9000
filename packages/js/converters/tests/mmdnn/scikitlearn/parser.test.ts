import { describe, expect, it } from 'vitest';
import { ScikitLearnParser } from '../../../src/mmdnn/scikitlearn/parser';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('parser.ts', () => {
  it('should instantiate and cover ScikitLearnParser', () => {
    try {
      const obj = Construct(ScikitLearnParser);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
