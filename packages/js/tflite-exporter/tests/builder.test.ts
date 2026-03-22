import { describe, it, expect } from 'vitest';
import { FlatBufferBuilder } from '../src/flatbuffer/builder';

describe('FlatBufferBuilder', () => {
  it('should create a basic flatbuffer object', () => {
    const builder = new FlatBufferBuilder();

    const strOffset = builder.createString('test');

    builder.startObject(2);
    builder.addFieldOffset(0, strOffset, 0);
    builder.addFieldInt32(1, 42, 0);
    const root = builder.endObject();

    builder.finish(root, 'TEST');
    const buf = builder.asUint8Array();

    expect(buf.length).toBeGreaterThan(0);
    expect(buf.length % 4).toBe(0);

    // Check magic bytes
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const magic = String.fromCharCode(
      view.getUint8(4),
      view.getUint8(5),
      view.getUint8(6),
      view.getUint8(7),
    );
    expect(magic).toBe('TEST');
  });
});
