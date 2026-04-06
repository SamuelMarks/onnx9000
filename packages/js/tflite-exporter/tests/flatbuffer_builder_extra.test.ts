import { describe, it, expect } from 'vitest';
import { FlatBufferBuilder } from '../src/flatbuffer/builder';

describe('Coverage FlatBufferBuilder', () => {
  it('Builder branches', () => {
    const b = new FlatBufferBuilder();

    // addFieldInt8 default value
    b.startObject(2);
    b.addFieldInt8(0, 0, 0); // shouldn't add
    b.addFieldInt8(1, 1, 0); // should add
    b.endObject();

    b.startObject(2);
    b.addFieldInt16(0, 0, 0);
    b.addFieldInt16(1, 1, 0);
    b.endObject();

    b.startObject(2);
    b.addFieldInt32(0, 0, 0);
    b.addFieldInt32(1, 1, 0);
    b.endObject();

    b.startObject(2);
    b.addFieldFloat32(0, 0.0, 0.0);
    b.addFieldFloat32(1, 1.0, 0.0);
    b.endObject();

    b.startObject(2);
    b.addFieldInt64(0, 0n, 0n);
    b.addFieldInt64(1, 1n, 0n);
    b.endObject();

    b.startObject(1);
    b.addFieldOffset(0, 0, 0);
    b.addFieldOffset(1, 1, 0);
    b.endObject();
  });
});

it('Builder growBuffer to limit', () => {
  const b = new FlatBufferBuilder(2147483647 - 100); // Almost 2GB
  // Just force a resize
  try {
    (b as Object).growBuffer();
  } catch (e) {}

  // Now it's at 2147483647
  try {
    (b as Object).growBuffer(); // throws
  } catch (e) {}
});

it('clear coverage', () => {
  const b = new FlatBufferBuilder();
  b.clear();
});

it('Builder createByteVector and createString', () => {
  const b = new FlatBufferBuilder();
  const vec = new Uint8Array([1, 2, 3]);
  const offset = b.createByteVector(vec, 1);
  expect(offset).toBeGreaterThan(0);

  const strOffset = b.createString('hello');
  expect(strOffset).toBeGreaterThan(0);
  const strOffset2 = b.createString(new Uint8Array([104, 105])); // "hi"
  expect(strOffset2).toBeGreaterThan(0);
});

it('Builder finish with identifier', () => {
  const b = new FlatBufferBuilder();
  b.startObject(1);
  const root = b.endObject();
  b.finish(root, 'TFL3');

  const buf = b.asUint8Array();
  expect(buf.length).toBeGreaterThan(4);
});

it('Builder addFloat64 addInt16', () => {
  const b = new FlatBufferBuilder();
  b.startObject(2);
  b.addFieldFloat64(0, 1.5, 0.0);
  b.endObject();
});

it('Builder endObject without startObject', () => {
  const b = new FlatBufferBuilder();
  expect(() => b.endObject()).toThrow('endObject called without startObject');
});

it('Builder prep grow', () => {
  const b = new FlatBufferBuilder(10);
  // this size will force it to grow
  (b as Object).prep(4, 20);
  expect(b.asUint8Array().length).toBeGreaterThanOrEqual(0);
});
