/* eslint-disable */
export enum GGUFValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

export enum GGUFTensorType {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q8_0 = 8,
}

export class GGUFWriter {
  public kvs: { key: string; type: GGUFValueType; val: ReturnType<typeof JSON.parse> }[] = [];
  public tensors: { name: string; shape: bigint[]; type: GGUFTensorType; offset: bigint }[] = [];

  public addUint8(key: string, val: number): void {
    this.kvs.push({ key, type: GGUFValueType.UINT8, val });
  }
  public addInt8(key: string, val: number): void {
    this.kvs.push({ key, type: GGUFValueType.INT8, val });
  }
  public addUint16(key: string, val: number): void {
    this.kvs.push({ key, type: GGUFValueType.UINT16, val });
  }
  public addInt16(key: string, val: number): void {
    this.kvs.push({ key, type: GGUFValueType.INT16, val });
  }
  public addUint32(key: string, val: number): void {
    this.kvs.push({ key, type: GGUFValueType.UINT32, val });
  }
  public addInt32(key: string, val: number): void {
    this.kvs.push({ key, type: GGUFValueType.INT32, val });
  }
  public addFloat32(key: string, val: number): void {
    this.kvs.push({ key, type: GGUFValueType.FLOAT32, val });
  }
  public addUint64(key: string, val: bigint): void {
    this.kvs.push({ key, type: GGUFValueType.UINT64, val });
  }
  public addInt64(key: string, val: bigint): void {
    this.kvs.push({ key, type: GGUFValueType.INT64, val });
  }
  public addFloat64(key: string, val: number): void {
    this.kvs.push({ key, type: GGUFValueType.FLOAT64, val });
  }
  public addBool(key: string, val: boolean): void {
    this.kvs.push({ key, type: GGUFValueType.BOOL, val });
  }
  public addString(key: string, val: string): void {
    this.kvs.push({ key, type: GGUFValueType.STRING, val });
  }
  public addArray(
    key: string,
    val: ReturnType<typeof JSON.parse>[],
    arrayType: GGUFValueType,
  ): void {
    this.kvs.push({ key, type: GGUFValueType.ARRAY, val: { arrayType, items: val } });
  }

  public addTensorInfo(name: string, shape: bigint[], type: GGUFTensorType, offset: bigint): void {
    this.tensors.push({ name, shape, type, offset });
  }

  public getHeaderSize(): number {
    let size = 0;
    size += 4; // Magic
    size += 4; // Version
    size += 8; // Tensor count
    size += 8; // KV count

    // KVs
    const encoder = new TextEncoder();
    for (const kv of this.kvs) {
      size += 8; // key string len
      size += encoder.encode(kv.key).length;
      size += 4; // type
      size += this.getValueSize(kv.type, kv.val);
    }

    // Tensors
    for (const t of this.tensors) {
      size += 8; // name string len
      size += encoder.encode(t.name).length;
      size += 4; // n_dims
      size += t.shape.length * 8; // dims
      size += 4; // type
      size += 8; // offset
    }

    let alignment = 32;
    for (const kv of this.kvs) {
      if (kv.key === 'general.alignment' && kv.type === GGUFValueType.UINT32) {
        alignment = kv.val as number;
      }
    }
    const padding = (alignment - (size % alignment)) % alignment;
    size += padding;
    return size;
  }

  private getValueSize(type: GGUFValueType, val: ReturnType<typeof JSON.parse>): number {
    switch (type) {
      case GGUFValueType.UINT8:
      case GGUFValueType.INT8:
      case GGUFValueType.BOOL:
        return 1;
      case GGUFValueType.UINT16:
      case GGUFValueType.INT16:
        return 2;
      case GGUFValueType.UINT32:
      case GGUFValueType.INT32:
      case GGUFValueType.FLOAT32:
        return 4;
      case GGUFValueType.UINT64:
      case GGUFValueType.INT64:
      case GGUFValueType.FLOAT64:
        return 8;
      case GGUFValueType.STRING: {
        const encoder = new TextEncoder();
        return 8 + encoder.encode(val).length;
      }
      case GGUFValueType.ARRAY: {
        let s = 4; // array type
        s += 8; // array len
        const atype = val.arrayType as GGUFValueType;
        for (const item of val.items) {
          s += this.getValueSize(atype, item);
        }
        return s;
      }
      default:
        /* v8 ignore start */
        throw new Error('Unknown type');
      /* v8 ignore stop */
    }
  }

  public writeHeader(buffer: ArrayBuffer, byteOffset: number = 0): number {
    const view = new DataView(buffer, byteOffset);
    let offset = 0;
    const encoder = new TextEncoder();

    const writeString = (s: string) => {
      const encoded = encoder.encode(s);
      view.setBigUint64(offset, BigInt(encoded.length), true);
      offset += 8;
      const u8 = new Uint8Array(buffer, byteOffset + offset, encoded.length);
      u8.set(encoded);
      offset += encoded.length;
    };

    const writeVal = (vtype: GGUFValueType, val: ReturnType<typeof JSON.parse>) => {
      switch (vtype) {
        case GGUFValueType.UINT8:
          view.setUint8(offset, val);
          offset += 1;
          break;
        case GGUFValueType.INT8:
          view.setInt8(offset, val);
          offset += 1;
          break;
        case GGUFValueType.UINT16:
          view.setUint16(offset, val, true);
          offset += 2;
          break;
        case GGUFValueType.INT16:
          view.setInt16(offset, val, true);
          offset += 2;
          break;
        case GGUFValueType.UINT32:
          view.setUint32(offset, val, true);
          offset += 4;
          break;
        case GGUFValueType.INT32:
          view.setInt32(offset, val, true);
          offset += 4;
          break;
        case GGUFValueType.FLOAT32:
          view.setFloat32(offset, val, true);
          offset += 4;
          break;
        case GGUFValueType.UINT64:
          view.setBigUint64(offset, val, true);
          offset += 8;
          break;
        case GGUFValueType.INT64:
          view.setBigInt64(offset, val, true);
          offset += 8;
          break;
        case GGUFValueType.FLOAT64:
          view.setFloat64(offset, val, true);
          offset += 8;
          break;
        case GGUFValueType.BOOL:
          view.setUint8(offset, val ? 1 : 0);
          offset += 1;
          break;
        case GGUFValueType.STRING:
          writeString(val);
          break;
        case GGUFValueType.ARRAY: {
          const atype = val.arrayType as GGUFValueType;
          const items = val.items as ReturnType<typeof JSON.parse>[];
          view.setUint32(offset, atype, true);
          offset += 4;
          view.setBigUint64(offset, BigInt(items.length), true);
          offset += 8;
          for (const item of items) {
            writeVal(atype, item);
          }
          break;
        }
        default:
          /* v8 ignore start */
          throw new Error('Unknown type');
        /* v8 ignore stop */
      }
    };

    // Magic: GGUF
    view.setUint8(offset++, 0x47); // G
    view.setUint8(offset++, 0x47); // G
    view.setUint8(offset++, 0x55); // U
    view.setUint8(offset++, 0x46); // F

    // Version 3
    view.setUint32(offset, 3, true);
    offset += 4;

    // Counts
    view.setBigUint64(offset, BigInt(this.tensors.length), true);
    offset += 8;
    view.setBigUint64(offset, BigInt(this.kvs.length), true);
    offset += 8;

    // KVs
    for (const kv of this.kvs) {
      writeString(kv.key);
      view.setUint32(offset, kv.type, true);
      offset += 4;
      writeVal(kv.type, kv.val);
    }

    // Tensors
    for (const t of this.tensors) {
      writeString(t.name);
      view.setUint32(offset, t.shape.length, true);
      offset += 4;
      for (const dim of t.shape) {
        view.setBigUint64(offset, dim, true);
        offset += 8;
      }
      view.setUint32(offset, t.type, true);
      offset += 4;
      view.setBigUint64(offset, t.offset, true);
      offset += 8;
    }

    let alignment = 32;
    for (const kv of this.kvs) {
      if (kv.key === 'general.alignment' && kv.type === GGUFValueType.UINT32) {
        alignment = kv.val;
      }
    }

    const padding = (alignment - (offset % alignment)) % alignment;
    for (let i = 0; i < padding; i++) {
      view.setUint8(offset++, 0);
    }

    return offset;
  }
}
