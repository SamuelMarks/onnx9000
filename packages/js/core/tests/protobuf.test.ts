import { describe, it, expect } from 'vitest';
import {
  BufferReader,
  BlobReader,
  readVarInt,
  readVarInt64,
  readString,
  readTag,
  skipField,
  WIRE_TYPE_VARINT,
  WIRE_TYPE_64BIT,
  WIRE_TYPE_LENGTH_DELIMITED,
  WIRE_TYPE_START_GROUP,
  WIRE_TYPE_END_GROUP,
  WIRE_TYPE_32BIT,
} from '../src/parser/protobuf.js';

describe('BufferReader', () => {
  it('should read bytes and skip', async () => {
    const data = new Uint8Array([1, 2, 3, 4, 5]);
    const reader = new BufferReader(data);
    expect(reader.getLength()).toBe(5);
    expect(reader.getPosition()).toBe(0);

    expect(await reader.readByte()).toBe(1);
    expect(reader.getPosition()).toBe(1);

    const bytes = await reader.readBytes(2);
    expect(bytes).toEqual(new Uint8Array([2, 3]));
    expect(reader.getPosition()).toBe(3);

    await reader.skip(1);
    expect(reader.getPosition()).toBe(4);

    expect(await reader.readByte()).toBe(5);

    await expect(reader.readByte()).rejects.toThrow('Unexpected end of buffer');
    await expect(reader.readBytes(1)).rejects.toThrow('Unexpected end of buffer');
  });
});

describe('BlobReader', () => {
  it('should read from Blob with cache', async () => {
    const data = new Uint8Array([10, 20, 30, 40, 50]);
    const blob = new Blob([data]);
    const reader = new BlobReader(blob, 2); // tiny cache size to force reload

    expect(reader.getLength()).toBe(5);
    expect(reader.getPosition()).toBe(0);

    expect(await reader.readByte()).toBe(10);
    expect(reader.getPosition()).toBe(1);

    const bytes = await reader.readBytes(3);
    expect(bytes).toEqual(new Uint8Array([20, 30, 40]));

    await reader.skip(1);
    expect(reader.getPosition()).toBe(5);

    await expect(reader.readByte()).rejects.toThrow('Unexpected end of blob');

    // reset to test readBytes over boundary
    const r2 = new BlobReader(blob, 2);
    await r2.skip(4);
    await expect(r2.readBytes(2)).rejects.toThrow('Unexpected end of blob');
  });
});

describe('Primitives', () => {
  it('should read varints', async () => {
    const data = new Uint8Array([0x01, 0x80, 0x01, 0xff, 0xff, 0xff, 0xff, 0x0f]);
    const reader = new BufferReader(data);
    expect(await readVarInt(reader)).toBe(1);
    expect(await readVarInt(reader)).toBe(128);
    expect((await readVarInt(reader)) >>> 0).toBe(0xffffffff);
  });

  it('should read varint64s', async () => {
    const data = new Uint8Array([0x01, 0x80, 0x01]);
    const reader = new BufferReader(data);
    expect(await readVarInt64(reader)).toBe(1n);
    expect(await readVarInt64(reader)).toBe(128n);
  });

  it('should read strings', async () => {
    const text = 'hello';
    const data = new TextEncoder().encode(text);
    const reader = new BufferReader(data);
    expect(await readString(reader, 5)).toBe(text);
  });

  it('should read tags', async () => {
    // tag 1, wire_type 2
    const data = new Uint8Array([(1 << 3) | 2]);
    const reader = new BufferReader(data);
    const tag = await readTag(reader);
    expect(tag.fieldNumber).toBe(1);
    expect(tag.wireType).toBe(2);
  });
});

describe('skipField', () => {
  it('should skip various wire types', async () => {
    const data = new Uint8Array([
      // varint
      0x01,
      // 64 bit
      0, 0, 0, 0, 0, 0, 0, 0,
      // length delimited
      2, 0, 0,
      // 32 bit
      0, 0, 0, 0,
    ]);
    const reader = new BufferReader(data);

    await skipField(reader, WIRE_TYPE_VARINT);
    expect(reader.getPosition()).toBe(1);

    await skipField(reader, WIRE_TYPE_64BIT);
    expect(reader.getPosition()).toBe(9);

    await skipField(reader, WIRE_TYPE_LENGTH_DELIMITED);
    expect(reader.getPosition()).toBe(12);

    await skipField(reader, WIRE_TYPE_32BIT);
    expect(reader.getPosition()).toBe(16);

    await expect(skipField(reader, WIRE_TYPE_START_GROUP)).rejects.toThrow(
      'Groups are not supported',
    );
    await expect(skipField(reader, WIRE_TYPE_END_GROUP)).rejects.toThrow(
      'Groups are not supported',
    );
    await expect(skipField(reader, 99)).rejects.toThrow('Unsupported wire type: 99');
  });
});
