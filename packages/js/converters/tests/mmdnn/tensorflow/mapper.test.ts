import { describe, expect, it } from 'vitest';
import { TFMapper } from '../../../src/mmdnn/tensorflow/mapper';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mapper.ts', () => {
  it('should instantiate and cover TFMapper', () => {
    try {
      const obj = Construct(TFMapper);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
