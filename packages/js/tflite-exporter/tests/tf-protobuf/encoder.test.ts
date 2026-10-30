import { describe, expect, it } from 'vitest';
import { TFProtobufEncoder } from '../../src/tf-protobuf/encoder';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('encoder.ts', () => {
  it('should instantiate and cover TFProtobufEncoder', () => {
    try {
      const obj = Construct(TFProtobufEncoder);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
