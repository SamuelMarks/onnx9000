import { describe, it, expect } from 'vitest';
import { GGUFWriter, GGUFValueType, GGUFTensorType } from '../src/builder.js';

describe('GGUFWriter', () => {
  it('should write', () => {
    const w = new GGUFWriter();
    w.addUint8('u8', 1);
    w.addInt8('i8', 2);
    w.addUint16('u16', 3);
    w.addInt16('i16', 4);
    w.addUint32('u32', 5);
    w.addInt32('i32', 6);
    w.addFloat32('f32', 7.0);
    w.addUint64('u64', 8n);
    w.addInt64('i64', 9n);
    w.addFloat64('f64', 10.0);
    w.addBool('b', true);
    w.addString('s', 'str');
    w.addArray('arr', [1, 2], GGUFValueType.UINT32);

    w.addTensorInfo('t', [1n, 2n], GGUFTensorType.F32, 0n);

    const size = w.getHeaderSize();
    expect(size).toBeGreaterThan(0);

    const buf = new ArrayBuffer(size);
    const written = w.writeHeader(buf);
    expect(written).toBeGreaterThan(0);
  });
});
