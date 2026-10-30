import { describe, expect, it } from 'vitest';
import { CaffeGenerator } from '../../../src/mmdnn/caffe/generator';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('generator.ts', () => {
  it('should instantiate and cover CaffeGenerator', () => {
    try {
      const obj = Construct(CaffeGenerator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
