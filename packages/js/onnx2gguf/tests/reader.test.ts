import { expect, test } from 'vitest';
import { GGUFWriter, GGUFValueType, GGUFTensorType } from '../src/builder';
import { GGUFReader } from '../src/reader';

test('GGUFReader basic', () => {
  const writer = new GGUFWriter();
  writer.addString('general.name', 'test');
  writer.addUint32('general.alignment', 32);
  writer.addArray('tokens', ['a', 'b'], GGUFValueType.STRING);
  writer.addTensorInfo('weight', [2n, 2n], GGUFTensorType.F32, 0n);
  writer.addTensorInfo('q4', [32n], GGUFTensorType.Q4_0, 32n);
  writer.addTensorInfo('q41', [32n], GGUFTensorType.Q4_1, 64n);
  writer.addTensorInfo('q8', [32n], GGUFTensorType.Q8_0, 96n);
  writer.addTensorInfo('f16', [2n], GGUFTensorType.F16, 160n);

  // Total size: header + 32 + 32 + 32 + 64 + 32
  const buf = new ArrayBuffer(1000);
  const off = writer.writeHeader(buf);

  const reader = new GGUFReader(buf);
  expect(reader.kvs['general.name']).toBe('test');
  expect(reader.kvs['tokens']).toEqual(['a', 'b']);

  const w = reader.getTensor('weight');
  expect(w.length).toBe(16);
  expect(reader.getTensor('q4').length).toBe(18);
  expect(reader.getTensor('q41').length).toBe(20);
  expect(reader.getTensor('q8').length).toBe(34);
  expect(reader.getTensor('f16').length).toBe(4);

  expect(() => reader.getTensor('missing')).toThrow('Tensor missing not found');
});

test('GGUFReader all types', () => {
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
  writer.addBool('b1', true);
  writer.addBool('b2', false);

  const buf = new ArrayBuffer(500);
  writer.writeHeader(buf);

  const reader = new GGUFReader(buf);
  expect(reader.kvs['u8']).toBe(1);
  expect(reader.kvs['i8']).toBe(-1);
  expect(reader.kvs['u16']).toBe(1);
  expect(reader.kvs['i16']).toBe(-1);
  expect(reader.kvs['u32']).toBe(1);
  expect(reader.kvs['i32']).toBe(-1);
  expect(reader.kvs['f32']).toBe(1.5);
  expect(reader.kvs['u64']).toBe(1n);
  expect(reader.kvs['i64']).toBe(-1n);
  expect(reader.kvs['f64']).toBe(1.5);
  expect(reader.kvs['b1']).toBe(true);
  expect(reader.kvs['b2']).toBe(false);
});

test('GGUFReader errors', () => {
  const b1 = new ArrayBuffer(4);
  new Uint8Array(b1).set([0x47, 0x47, 0x55, 0x58]);
  expect(() => new GGUFReader(b1)).toThrow('Not a GGUF file');

  const b2 = new ArrayBuffer(8);
  new Uint8Array(b2).set([0x47, 0x47, 0x55, 0x46, 4, 0, 0, 0]);
  expect(() => new GGUFReader(b2)).toThrow('Unsupported GGUF version 4');
});
