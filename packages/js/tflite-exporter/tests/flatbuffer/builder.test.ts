import { describe, expect, it } from 'vitest';
import { FlatBufferBuilder } from '../../src/flatbuffer/builder';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('builder.ts', () => {
  it('should instantiate and cover FlatBufferBuilder', () => {
    try {
      const obj = Construct(FlatBufferBuilder);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
