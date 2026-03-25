import { expect, test } from 'vitest';
import { GGUFWriter, GGUFValueType, GGUFTensorType } from '../src/builder';

test('GGUFWriter basic', () => {
  const writer = new GGUFWriter();
  writer.addString('general.name', 'test');
  writer.addUint32('general.alignment', 32);
  writer.addArray('tokens', ['a', 'b'], GGUFValueType.STRING);
  writer.addTensorInfo('weight', [2n, 2n], GGUFTensorType.F32, 0n);

  const size = writer.getHeaderSize();
  const buffer = new ArrayBuffer(size);
  const written = writer.writeHeader(buffer);
  expect(written).toBe(size);

  const view = new DataView(buffer);
  expect(view.getUint8(0)).toBe(0x47);
  expect(view.getUint8(1)).toBe(0x47);
  expect(view.getUint8(2)).toBe(0x55);
  expect(view.getUint8(3)).toBe(0x46);
  expect(view.getUint32(4, true)).toBe(3);
  expect(view.getBigUint64(8, true)).toBe(1n); // tensor count
  expect(view.getBigUint64(16, true)).toBe(3n); // KV count
});

test('GGUFWriter all types', () => {
  const writer = new GGUFWriter();
  writer.addUint8('u8', 1);
  writer.addInt8('i8', -1);
  writer.addUint16('u16', 1);
  writer.addInt16('i16', -1);
  writer.addUint32('u32', 1);
  writer.addInt32('i32', -1);
  writer.addFloat32('f32', 1.5);
  writer.addUint64('u64', 1n);
  writer.addInt64('i64', -1n);
  writer.addFloat64('f64', 1.5);
  writer.addBool('bool', true);
  writer.addBool('bool2', false);

  const size = writer.getHeaderSize();
  const buffer = new ArrayBuffer(size);
  const written = writer.writeHeader(buffer);
  expect(written).toBe(size);
});
