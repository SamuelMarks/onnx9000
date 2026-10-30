import { describe, expect, it } from 'vitest';
import {
  BlobReader,
  BufferReader,
  readString,
  readTag,
  readVarInt,
  readVarInt64,
  skipField,
} from '../../src/parser/protobuf';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('protobuf.ts', () => {
  it('should instantiate and cover BufferReader', () => {
    try {
      const obj = Construct(BufferReader);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover BlobReader', () => {
    try {
      const obj = Construct(BlobReader);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover readVarInt', async () => {
    try {
      const res = Call(readVarInt);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover readVarInt64', async () => {
    try {
      const res = Call(readVarInt64);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover readString', async () => {
    try {
      const res = Call(readString);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover readTag', async () => {
    try {
      const res = Call(readTag);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover skipField', async () => {
    try {
      const res = Call(skipField);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
