import { BuiltinOperator } from '../src/flatbuffer/schema';
import { describe, it, expect } from 'vitest';
import { FlatBufferReader } from '../src/flatbuffer/reader';
import { FlatBufferBuilder } from '../src/flatbuffer/builder';
import { Tensor as SchemaTensor, TensorType } from '../src/flatbuffer/schema';
import { Operator as SchemaOperator, BuiltinOptions } from '../src/flatbuffer/schema';
import {
  OperatorCode,
  QuantizationParameters,
  SubGraph,
  Buffer,
  Model,
  Metadata,
} from '../src/flatbuffer/schema';
import { TFLiteExporter } from '../src/exporter';

describe('Coverage FlatBuffer', () => {
  it('Reader edges', () => {
    const shortBuf = new Uint8Array([1, 2, 3]);
    const reader1 = new FlatBufferReader(shortBuf);
    expect(reader1.checkMagicBytes('TFL3')).toBe(false);

    const buf = new Uint8Array(100);
    const view = new DataView(buf.buffer);
    const reader2 = new FlatBufferReader(buf);

    view.setInt32(20, -10, true);
    view.setInt16(10, 6, true);
    view.setInt16(12, 10, true);
    view.setInt16(10 + 4, 0, true);

    expect(reader2.getString(20, 0)).toBeNull();
  });

  it('Schema edges', () => {
    const builder = new FlatBufferBuilder(1024);
    builder.startObject(10); // arbitrary
    const tensorLoc = SchemaTensor.create(
      builder,
      0,
      TensorType.FLOAT32,
      0,
      0,
      0,
      true,
      0,
      0,
      true,
    );
    expect(tensorLoc).toBeGreaterThan(0);

    builder.startObject(10);
    const opLoc = SchemaOperator.create(
      builder,
      0,
      0,
      0,
      BuiltinOptions.AddOptions,
      0,
      0,
      0,
      true,
      0,
    );
    expect(opLoc).toBeGreaterThan(0);

    builder.startObject(4);
    const opCodeLoc = OperatorCode.create(builder, 0, 0, 0, 0);
    expect(opCodeLoc).toBeGreaterThan(0);
  });

  it('Reader int getter fallbacks', () => {
    const buf = new Uint8Array(100);
    const view = new DataView(buf.buffer);
    const reader = new FlatBufferReader(buf);

    view.setInt32(20, -10, true);
    view.setInt16(10, 6, true);
    view.setInt16(12, 10, true);
    view.setInt16(10 + 4, 0, true);

    // Test getInt8, getInt16, getInt32 defaults where field offset is 0
    expect(reader.getInt8(20, 0, 99)).toBe(99);
    expect(reader.getInt16(20, 0, 99)).toBe(99);
    expect(reader.getInt32(20, 0, 99)).toBe(99);
  });

  it('Reader getFloat32 coverage', () => {
    const buf = new Uint8Array(100);
    const view = new DataView(buf.buffer);
    const reader = new FlatBufferReader(buf);

    view.setInt32(20, -10, true);
    view.setInt16(10, 6, true);
    view.setInt16(12, 10, true);
    view.setInt16(10 + 4, 0, true);

    expect(reader.getFloat32(20, 0, 1.5)).toBe(1.5);
  });

  it('Schema more edges', () => {
    const builder = new FlatBufferBuilder(1024);
    builder.startObject(7);
    const qLoc = QuantizationParameters.create(builder, 0, 0, 0, 0, 0, 0, 0);
    expect(qLoc).toBeGreaterThan(0);

    builder.startObject(5);
    const subLoc = SubGraph.create(builder, 0, 0, 0, 0, 0);
    expect(subLoc).toBeGreaterThan(0);

    builder.startObject(1);
    const bufLoc = Buffer.create(builder, 0);
    expect(bufLoc).toBeGreaterThan(0);

    builder.startObject(6);
    const mLoc = Model.create(builder, 0, 0, 0, 0, 0, 0);
    expect(mLoc).toBeGreaterThan(0);
  });

  it('Builder growBuffer from 0 and finish without identifier', () => {
    const b = new FlatBufferBuilder(0);
    (b as Object).growBuffer();
    expect((b as Object).bb.length).toBeGreaterThan(0);

    const b2 = new FlatBufferBuilder(100);
    b2.startObject(1);
    const root = b2.endObject();
    b2.finish(root); // no identifier
  });

  it('Schema Metadata', () => {
    const b = new FlatBufferBuilder(100);
    b.startObject(2);
    const mLoc = Metadata.create(b, 0, 0);
    expect(mLoc).toBeGreaterThan(0);

    // Hit 372-375
    b.startObject(10);
    const t2Loc = SchemaTensor.create(b, 0, TensorType.FLOAT32, 0, 0, 0, false, 0, 0, false);
    expect(t2Loc).toBeGreaterThan(0);

    // Hit 402
    b.startObject(10);
    const op2Loc = SchemaOperator.create(b, 0, 0, 0, BuiltinOptions.AddOptions, 0, 0, 0, false, 0);
    expect(op2Loc).toBeGreaterThan(0);
  });

  it('Reader getRoot and getRootAsModel and checkMagicBytes true', () => {
    const b = new FlatBufferBuilder(100);
    b.startObject(1);
    const root = b.endObject();
    b.finish(root, 'TFL3');

    const r = new FlatBufferReader(b.asUint8Array());
    expect(r.getRoot()).toBeGreaterThan(0);
    expect(r.getRootAsModel()).toBeGreaterThan(0);
    expect(r.checkMagicBytes('TFL3')).toBe(true);

    // Hit 45 in reader
    const buf = new Uint8Array(100);
    const view = new DataView(buf.buffer);
    view.setInt32(20, -10, true);
    view.setInt16(10, 6, true);
    view.setInt16(12, 10, true);
    view.setInt16(10 + 4, 0, true);
    const r2 = new FlatBufferReader(buf);
    expect(r2.getFieldOffset(20, 0)).toBe(0);
  });

  it('Exporter edges', () => {
    const exp = new TFLiteExporter();

    // addTensorBufferLazily
    exp.addTensorBufferLazily([1, 2, 3, 4, 5, 6, 7], 10, () => new Uint8Array(10));

    expect(() => exp.addTensorBufferLazily([1], 2 ** 31, () => new Uint8Array(1))).toThrow(
      'limits (size:',
    );

    process.env['TFLITE_MEDIAPIPE_METADATA'] = '1';
    exp.finish(0); // This calls finish which wraps Model.create
    delete process.env['TFLITE_MEDIAPIPE_METADATA'];

    expect(exp.toJSON().version).toBe(3);

    // Hit empty toJSON
    const exp2 = new TFLiteExporter();
    exp2.toJSON();

    exp.destroy();
  });
});

it('Exporter more edges', () => {
  // ('../src/exporter');
  // ('../src/flatbuffer/schema');
  const exp = new TFLiteExporter();

  exp.addMetadata('test-meta', new Uint8Array([1]));

  exp.addBuffer(new Uint8Array(0)); // empty cache
  const b1 = exp.addBuffer(new Uint8Array([1]));
  const b2 = exp.addBuffer(new Uint8Array([1])); // hash cache
  expect(b1).toBe(b2);

  const op1 = exp.getOrAddOperatorCode(BuiltinOperator.ADD);
  const op2 = exp.getOrAddOperatorCode(BuiltinOperator.TRANSPOSE_CONV);
  const op3 = exp.getOrAddOperatorCode(BuiltinOperator.RESIZE_BILINEAR);
  const op4 = exp.getOrAddOperatorCode(BuiltinOperator.ADD, 'custom', 5);
  const op5 = exp.getOrAddOperatorCode(BuiltinOperator.ADD, 'custom', 5); // hash cache

  exp.finish(0); // with metadata
});
