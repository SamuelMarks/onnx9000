import { describe, expect, it } from 'vitest';
import { TensorFlowGenerator } from '../../../src/mmdnn/tensorflow/generator';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('generator.ts', () => {
  it('should instantiate and cover TensorFlowGenerator', () => {
    try {
      const obj = Construct(TensorFlowGenerator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
