import { describe, it, expect } from 'vitest';
import { parseModelProto } from '../src/parser/onnx';
import { BufferReader } from '../src/parser/protobuf';

// Minimal protobuf writer to generate specific bytes for testing
class ProtoWriter {
  buffers: Uint8Array[] = [];

  writeVarInt(value: number) {
    let temp = value;
    const bytes: number[] = [];
    do {
      let b = temp & 0x7f;
      temp >>>= 7;
      if (temp !== 0) {
        b |= 0x80;
      }
      bytes.push(b);
    } while (temp !== 0);
    this.buffers.push(new Uint8Array(bytes));
  }

  writeTag(fieldNumber: number, wireType: number) {
    this.writeVarInt((fieldNumber << 3) | wireType);
  }

  writeString(str: string) {
    const encoder = new TextEncoder();
    const bytes = encoder.encode(str);
    this.writeVarInt(bytes.length);
    this.buffers.push(bytes);
  }

  writeBytes(bytes: Uint8Array) {
    this.writeVarInt(bytes.length);
    this.buffers.push(bytes);
  }

  writeFloat32(val: number) {
    const buf = new Uint8Array(4);
    new DataView(buf.buffer).setFloat32(0, val, true);
    this.buffers.push(buf);
  }

  build(): Uint8Array {
    const totalLen = this.buffers.reduce((acc, b) => acc + b.length, 0);
    const result = new Uint8Array(totalLen);
    let offset = 0;
    for (const b of this.buffers) {
      result.set(b, offset);
      offset += b.length;
    }
    return result;
  }
}

describe('ONNX Parser Coverage Gaps', () => {
  it('handles unknown field in opsetImports (106, 107)', async () => {
    const w = new ProtoWriter();

    const opsetW = new ProtoWriter();
    opsetW.writeTag(1, 2); // domain
    opsetW.writeString('test_domain');
    opsetW.writeTag(3, 2); // unknown field, wireType 2
    opsetW.writeString('unknown_data');
    opsetW.writeTag(2, 0); // version
    opsetW.writeVarInt(1);

    w.writeTag(8, 2); // opset_import
    w.writeBytes(opsetW.build());

    const reader = new BufferReader(w.build());
    const g = await parseModelProto(reader);
    expect(g.opsetImports['test_domain']).toBe(1);
  });

  it('handles modelVersion, docString and unknown field (123, 127, 131)', async () => {
    const w = new ProtoWriter();
    w.writeTag(5, 0); // model_version
    w.writeVarInt(2);
    w.writeTag(6, 2); // doc_string
    w.writeString('doc');
    w.writeTag(99, 2); // unknown
    w.writeString('junk');

    const reader = new BufferReader(w.build());
    const g = await parseModelProto(reader);
    expect(g.modelVersion).toBe(2);
    expect(g.docString).toBe('doc');
  });

  it('handles corrupted model data (134-138)', async () => {
    const w = new ProtoWriter();
    w.writeTag(1, 0);
    // Truncate
    const reader = new BufferReader(w.build().subarray(0, 1));
    await parseModelProto(reader);
    // Should warn, not throw
  });

  it('handles corrupted graph data (178-180)', async () => {
    const w = new ProtoWriter();
    const gW = new ProtoWriter();
    gW.writeTag(1, 2); // node but cut it
    w.writeTag(7, 2); // graph
    w.writeBytes(gW.build().subarray(0, 1));
    const reader = new BufferReader(w.build());
    await parseModelProto(reader);
  });

  it('handles unknown node fields (224-225)', async () => {
    const w = new ProtoWriter();
    const gW = new ProtoWriter();
    const nW = new ProtoWriter();
    nW.writeTag(4, 2); // op_type
    nW.writeString('Relu');
    nW.writeTag(99, 2); // unknown
    nW.writeString('junk');

    gW.writeTag(1, 2); // node
    gW.writeBytes(nW.build());

    w.writeTag(7, 2); // graph
    w.writeBytes(gW.build());

    const reader = new BufferReader(w.build());
    const g = await parseModelProto(reader);
    expect(g.nodes[0].opType).toBe('Relu');
  });

  it('handles dim_param and unknown shape fields (275-280, 284, 288, 292, 297)', async () => {
    const w = new ProtoWriter();
    const gW = new ProtoWriter();
    const vW = new ProtoWriter();
    vW.writeTag(1, 2); // name
    vW.writeString('in1');

    const tTypeW = new ProtoWriter();
    const tShapeW = new ProtoWriter();

    const dimW = new ProtoWriter();
    dimW.writeTag(2, 2); // dim_param
    dimW.writeString('N');
    dimW.writeTag(99, 2); // unknown dim field
    dimW.writeString('junk');

    tShapeW.writeTag(1, 2); // dim
    tShapeW.writeBytes(dimW.build());
    tShapeW.writeTag(99, 2); // unknown shape field
    tShapeW.writeString('junk');

    tTypeW.writeTag(1, 0); // elem_type
    tTypeW.writeVarInt(1); // float32
    tTypeW.writeTag(2, 2); // shape
    tTypeW.writeBytes(tShapeW.build());
    tTypeW.writeTag(99, 2); // unknown tensor type field
    tTypeW.writeString('junk');

    const typeW = new ProtoWriter();
    typeW.writeTag(1, 2); // tensor_type
    typeW.writeBytes(tTypeW.build());
    typeW.writeTag(99, 2); // unknown type field
    typeW.writeString('junk');

    vW.writeTag(2, 2); // type
    vW.writeBytes(typeW.build());
    vW.writeTag(99, 2); // unknown value info field
    vW.writeString('junk');

    gW.writeTag(11, 2); // input
    gW.writeBytes(vW.build());

    w.writeTag(7, 2); // graph
    w.writeBytes(gW.build());

    const reader = new BufferReader(w.build());
    const g = await parseModelProto(reader);
    expect(g.inputs[0].shape[0]).toBe('N');
  });

  it('handles floats attribute (352-366) and ints attribute (371-380)', async () => {
    const w = new ProtoWriter();
    const gW = new ProtoWriter();
    const nW = new ProtoWriter();
    nW.writeTag(4, 2); // op_type
    nW.writeString('Relu');

    const attrW1 = new ProtoWriter(); // floats list (length delimited)
    attrW1.writeTag(1, 2);
    attrW1.writeString('attr_f_list');
    attrW1.writeTag(20, 0);
    attrW1.writeVarInt(6); // FLOATS
    const floatsW = new ProtoWriter();
    floatsW.writeFloat32(1.5);
    floatsW.writeFloat32(2.5);
    attrW1.writeTag(7, 2);
    attrW1.writeBytes(floatsW.build());

    const attrW2 = new ProtoWriter(); // floats (32-bit elements, wire type 5)
    attrW2.writeTag(1, 2);
    attrW2.writeString('attr_f_32');
    attrW2.writeTag(20, 0);
    attrW2.writeVarInt(6); // FLOATS
    attrW2.writeTag(7, 5);
    attrW2.writeFloat32(3.5);
    attrW2.writeTag(7, 5);
    attrW2.writeFloat32(4.5);

    const attrW3 = new ProtoWriter(); // ints list (length delimited)
    attrW3.writeTag(1, 2);
    attrW3.writeString('attr_i_list');
    attrW3.writeTag(20, 0);
    attrW3.writeVarInt(7); // INTS
    const intsW = new ProtoWriter();
    intsW.writeVarInt(10);
    intsW.writeVarInt(20);
    attrW3.writeTag(8, 2);
    attrW3.writeBytes(intsW.build());

    const attrW4 = new ProtoWriter(); // ints (varint elements, wire type 0)
    attrW4.writeTag(1, 2);
    attrW4.writeString('attr_i_0');
    attrW4.writeTag(20, 0);
    attrW4.writeVarInt(7); // INTS
    attrW4.writeTag(8, 0);
    attrW4.writeVarInt(30);
    attrW4.writeTag(8, 0);
    attrW4.writeVarInt(40);

    nW.writeTag(5, 2);
    nW.writeBytes(attrW1.build());
    nW.writeTag(5, 2);
    nW.writeBytes(attrW2.build());
    nW.writeTag(5, 2);
    nW.writeBytes(attrW3.build());
    nW.writeTag(5, 2);
    nW.writeBytes(attrW4.build());

    gW.writeTag(1, 2); // node
    gW.writeBytes(nW.build());
    w.writeTag(7, 2); // graph
    w.writeBytes(gW.build());

    const reader = new BufferReader(w.build());
    const g = await parseModelProto(reader);
    const attrs = g.nodes[0].attributes;
    if (!attrs['attr_f_list']) throw new Error('missing attr_f_list');
    expect(attrs['attr_f_list'].value).toEqual([1.5, 2.5]);
    expect(attrs['attr_f_32'].value).toEqual([3.5, 4.5]);
    expect(attrs['attr_i_list'].value).toEqual([10, 20]);
    expect(attrs['attr_i_0'].value).toEqual([30, 40]);
  });

  it('handles unknown attribute type (407-412)', async () => {
    const w = new ProtoWriter();
    const gW = new ProtoWriter();
    const nW = new ProtoWriter();
    nW.writeTag(4, 2);
    nW.writeString('Relu');
    const attrW = new ProtoWriter();
    attrW.writeTag(1, 2);
    attrW.writeString('attr_unk');
    attrW.writeTag(99, 2);
    attrW.writeString('junk'); // unknown field in attr
    nW.writeTag(5, 2);
    nW.writeBytes(attrW.build());
    gW.writeTag(1, 2);
    gW.writeBytes(nW.build());
    w.writeTag(7, 2);
    w.writeBytes(gW.build());
    const reader = new BufferReader(w.build());
    await parseModelProto(reader);
  });

  it('handles tensor types and raw data (435-440, 452-455, 474-475)', async () => {
    const w = new ProtoWriter();
    const gW = new ProtoWriter();
    const tW = new ProtoWriter();

    tW.writeTag(8, 2);
    tW.writeString('t1');
    tW.writeTag(2, 0);
    tW.writeVarInt(16); // dataType = bfloat16

    // dims length delimited
    const dimsW = new ProtoWriter();
    dimsW.writeVarInt(2);
    dimsW.writeVarInt(3);
    tW.writeTag(1, 2);
    tW.writeBytes(dimsW.build());

    // dims varint
    tW.writeTag(1, 0);
    tW.writeVarInt(4);

    // raw data
    tW.writeTag(9, 2);
    tW.writeString('raw');

    // external data
    const extW = new ProtoWriter();
    extW.writeTag(1, 2);
    extW.writeString('location');
    extW.writeTag(2, 2);
    extW.writeString('data.bin');
    extW.writeTag(99, 2);
    extW.writeString('junk'); // unknown field
    tW.writeTag(13, 2);
    tW.writeBytes(extW.build());

    tW.writeTag(14, 0);
    tW.writeVarInt(1); // data_location

    gW.writeTag(5, 2); // initializer
    gW.writeBytes(tW.build());
    w.writeTag(7, 2);
    w.writeBytes(gW.build());

    const reader = new BufferReader(w.build());
    const g = await parseModelProto(reader);
    const t = g.tensors[g.initializers[0]];
    expect(t.name).toBe('t1');
    expect(t.shape).toEqual([2, 3, 4]);
    expect(t.dtype).toBe('bfloat16');
    expect(t.externalData!.location).toBe('data.bin');
  });

  it('covers remaining tensor datatypes', async () => {
    const types = [
      { t: 2, s: 'uint8' },
      { t: 3, s: 'int8' },
      { t: 4, s: 'uint16' },
      { t: 5, s: 'int16' },
      { t: 9, s: 'bool' },
      { t: 10, s: 'float16' },
      { t: 11, s: 'float64' },
      { t: 12, s: 'uint32' },
      { t: 13, s: 'uint64' },
      { t: 16, s: 'bfloat16' },
    ];

    for (const type of types) {
      const w = new ProtoWriter();
      const gW = new ProtoWriter();
      const tW = new ProtoWriter();
      tW.writeTag(8, 2);
      tW.writeString(`t_${type.s}`);
      tW.writeTag(2, 0);
      tW.writeVarInt(type.t);
      gW.writeTag(5, 2);
      gW.writeBytes(tW.build());
      w.writeTag(7, 2);
      w.writeBytes(gW.build());

      const reader = new BufferReader(w.build());
      const g = await parseModelProto(reader);
      expect(g.tensors[g.initializers[0]].dtype).toBe(type.s);
    }
  });

  it('throws error on unsupported tensor type', async () => {
    const w = new ProtoWriter();
    const gW = new ProtoWriter();
    const tW = new ProtoWriter();
    tW.writeTag(8, 2);
    tW.writeString('bad_t');
    tW.writeTag(2, 0);
    tW.writeVarInt(999);
    gW.writeTag(5, 2);
    gW.writeBytes(tW.build());
    w.writeTag(7, 2);
    w.writeBytes(gW.build());

    const reader = new BufferReader(w.build());
    await parseModelProto(reader);
  });

  it('covers remaining attribute types', async () => {
    const w = new ProtoWriter();
    const gW = new ProtoWriter();
    const nW = new ProtoWriter();
    nW.writeTag(4, 2);
    nW.writeString('Test');

    const addAttr = (typeNum: number, name: string) => {
      const aW = new ProtoWriter();
      aW.writeTag(1, 2);
      aW.writeString(name);
      aW.writeTag(20, 0);
      aW.writeVarInt(typeNum);
      return aW;
    };

    const types = [
      { t: 9, n: 'TENSORS' },
      { t: 10, n: 'GRAPHS' },
      { t: 11, n: 'SPARSE_TENSOR' },
      { t: 12, n: 'SPARSE_TENSORS' },
      { t: 99, n: 'UNKNOWN' },
    ];

    for (const type of types) {
      nW.writeTag(5, 2);
      nW.writeBytes(addAttr(type.t, type.n).build());
    }

    gW.writeTag(1, 2);
    gW.writeBytes(nW.build());
    w.writeTag(7, 2);
    w.writeBytes(gW.build());

    const reader = new BufferReader(w.build());
    const g = await parseModelProto(reader);
    const attrs = g.nodes[0].attributes;
    expect(attrs['TENSORS'].type).toBe('TENSORS');
    expect(attrs['GRAPHS'].type).toBe('GRAPHS');
    expect(attrs['SPARSE_TENSOR'].type).toBe('SPARSE_TENSOR');
    expect(attrs['SPARSE_TENSORS'].type).toBe('SPARSE_TENSORS');
    expect(attrs['UNKNOWN'].type).toBe('UNKNOWN');
  });
});

it('handles externalDataMap location fallback', async () => {
  const w = new ProtoWriter();
  const gW = new ProtoWriter();
  const tW = new ProtoWriter();
  tW.writeTag(8, 2);
  tW.writeString('t2');
  tW.writeTag(2, 0);
  tW.writeVarInt(1);
  tW.writeTag(14, 0);
  tW.writeVarInt(1); // dataLocation = 1 but no location key
  gW.writeTag(5, 2);
  gW.writeBytes(tW.build());
  w.writeTag(7, 2);
  w.writeBytes(gW.build());
  const reader = new BufferReader(w.build());
  const g = await parseModelProto(reader);
  expect(g.tensors[g.initializers[0]].externalData!.location).toBe('');
});
