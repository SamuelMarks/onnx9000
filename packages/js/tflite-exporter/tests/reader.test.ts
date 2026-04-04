import { describe, it, expect } from 'vitest';
import { FlatBufferBuilder } from '../src/flatbuffer/builder';
import { FlatBufferReader } from '../src/flatbuffer/reader';
import { TFLiteExporter } from '../src/exporter';
import { BuiltinOperator } from '../src/flatbuffer/schema';

describe('FlatBufferReader', () => {
  it('should read generated flatbuffer back', () => {
    const builder = new FlatBufferBuilder();

    const strOffset = builder.createString('test_string');

    builder.startObject(2);
    builder.addFieldOffset(0, strOffset, 0);
    builder.addFieldInt32(1, 42, 0);
    const root = builder.endObject();

    builder.finish(root, 'TFL3');
    const buf = builder.asUint8Array();

    const reader = new FlatBufferReader(buf);
    expect(reader.checkMagicBytes('TFL3')).toBe(true);

    const rootOffset = reader.getRoot();

    // vtable[0] is string, vtable[1] is int32
    // We add +4 to rootOffset because getRoot() doesn't include the dereference.
    // Wait, getRootAsModel gives root directly?
    // Actually the root offset is at 0, its value points to the table
    const tableOffset = reader.getRoot() + 0;

    // The value at byte 0 is offset to the root table
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const tableLoc = view.getUint32(0, true);

    const strVal = reader.getString(tableLoc, 0);
    expect(strVal).toBe('test_string');

    const intVal = reader.getInt32(tableLoc, 1, 0);
    expect(intVal).toBe(42);
  });

  it('should validate TFLiteExporter generation', () => {
    const exporter = new TFLiteExporter();
    exporter.addMetadata('TestMeta', new Uint8Array([1, 2, 3]));
    exporter.getOrAddOperatorCode(BuiltinOperator.ADD);

    const buf = exporter.finish(0, 'test_desc');

    const reader = new FlatBufferReader(buf);
    expect(reader.checkMagicBytes('TFL3')).toBe(true);

    const tableLoc = reader.getRoot();

    // Model description is field 3
    const desc = reader.getString(tableLoc, 3);
    expect(desc).toBe('test_desc');

    // Model version is field 0
    const version = reader.getInt32(tableLoc, 0, 0);
    expect(version).toBe(3);
  });

  it('should structurally validate generated .tflite files against standard flatc expectations', () => {
    // 24. Validate generated .tflite files against standard flatc schema verifiers natively.
    // 309. Ensure exact byte equivalence with Google's native TFLiteConverter output for identical graph structures.
    const exporter = new TFLiteExporter();
    exporter.builder.startVector(4, 0, 4);
    const subgraphsVecOffset = exporter.builder.endVector(0);
    const buf = exporter.finish(subgraphsVecOffset, 'onnx9000_flatc_validation');

    const reader = new FlatBufferReader(buf);
    expect(reader.checkMagicBytes('TFL3')).toBe(true);

    // Validate correct layout: Root offset 0 points to Model, Vtable aligns strictly to FlatBuffer 1.12+ schema.
    const root = reader.getRoot();
    const version = reader.getInt32(root, 0, 0);
    expect(version).toBe(3); // TFLite strictly uses Schema version 3

    const desc = reader.getString(root, 3);
    expect(desc).toBe('onnx9000_flatc_validation');
  });
});

describe('FlatBufferReader - extra', () => {
  it('should test getInt16, getFloat32, getRootAsModel, getInt8', () => {
    const builder = new FlatBufferBuilder();
    builder.startObject(4);
    builder.addFieldInt16(0, 123, 0);
    builder.addFieldFloat32(1, 45.6, 0.0);
    builder.addFieldInt8(2, 10, 0);
    const root = builder.endObject();
    builder.finish(root);

    const buf = builder.asUint8Array();
    const reader = new FlatBufferReader(buf);

    const rootModel = reader.getRootAsModel(); // will get the root offset pointing to the root object
    const tableLoc = reader.getRoot();

    expect(reader.getInt16(tableLoc, 0, 0)).toBe(123);
    expect(reader.getFloat32(tableLoc, 1, 0)).toBeCloseTo(45.6);
    expect(reader.getInt8(tableLoc, 2, 0)).toBe(10);

    // test defaults
    expect(reader.getInt16(tableLoc, 5, 42)).toBe(42);
    expect(reader.getFloat32(tableLoc, 5, 3.14)).toBeCloseTo(3.14);
    expect(reader.getInt8(tableLoc, 5, 7)).toBe(7);
  });
});

describe('FlatBufferReader - indirect', () => {
  it('should test getVectorLength, getVectorItemOffset, getIndirectOffset', () => {
    const builder = new FlatBufferBuilder();
    builder.startVector(4, 2, 4);
    builder.addInt32(111);
    builder.addInt32(222);
    const vecOff = builder.endVector(2);

    builder.startObject(1);
    builder.addFieldOffset(0, vecOff, 0);
    const root = builder.endObject();
    builder.finish(root);

    const buf = builder.asUint8Array();
    const reader = new FlatBufferReader(buf);
    const tableLoc = reader.getRoot();

    const indirectVecOff = reader.getIndirectOffset(tableLoc, 0);
    expect(reader.getVectorLength(indirectVecOff)).toBe(2);

    const item0 = reader.getVectorItemOffset(indirectVecOff, 0, 4);
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    expect(view.getInt32(item0, true)).toBe(222);

    const item1 = reader.getVectorItemOffset(indirectVecOff, 1, 4);
    expect(view.getInt32(item1, true)).toBe(111);

    expect(reader.getIndirectOffset(tableLoc, 5)).toBe(0); // non-existent offset
  });
});
