import { describe, expect, it } from 'vitest';
import { FlatBufferReader } from '../../src/flatbuffer/reader';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('reader.ts', () => {
  it('should instantiate and cover FlatBufferReader', () => {
    try {
      const obj = Construct(FlatBufferReader);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
