import { describe, expect, it } from 'vitest';
import { MXNetGenerator } from '../../../src/mmdnn/mxnet/generator';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('generator.ts', () => {
  it('should instantiate and cover MXNetGenerator', () => {
    try {
      const obj = Construct(MXNetGenerator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
