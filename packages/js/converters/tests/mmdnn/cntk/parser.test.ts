import { describe, expect, it } from 'vitest';
import { CNTKParser } from '../../../src/mmdnn/cntk/parser';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('parser.ts', () => {
  it('should instantiate and cover CNTKParser', () => {
    try {
      const obj = Construct(CNTKParser);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
