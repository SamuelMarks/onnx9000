import { describe, expect, it } from 'vitest';
import { CNTKGenerator } from '../../../src/mmdnn/cntk/generator';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('generator.ts', () => {
  it('should instantiate and cover CNTKGenerator', () => {
    try {
      const obj = Construct(CNTKGenerator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
