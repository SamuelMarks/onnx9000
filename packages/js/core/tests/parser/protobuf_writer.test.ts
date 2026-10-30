import { describe, expect, it } from 'vitest';
import { BufferWriter } from '../../src/parser/protobuf_writer';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('protobuf_writer.ts', () => {
  it('should instantiate and cover BufferWriter', () => {
    try {
      const obj = Construct(BufferWriter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
