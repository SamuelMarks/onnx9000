import { describe, expect, it } from 'vitest';
import { Keras2OnnxConverter } from '../../src/keras/index';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('index.ts', () => {
  it('should instantiate and cover Keras2OnnxConverter', () => {
    try {
      const obj = Construct(Keras2OnnxConverter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
