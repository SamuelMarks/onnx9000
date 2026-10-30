/**
 * @fileoverview reader.ts
 * Provides reader functionality for the onnx2gguf package.
 */
import { Onnx9000Error } from '@onnx9000/core';
import { GGUFTensorType, GGUFValueType } from './builder';

/** Represents the `GGUFReader` class. */
export class GGUFReader {
  /** Property `kvs` of `GGUFReader`. */
  public kvs: Record<string, ReturnType<typeof JSON.parse>> = {};
  /** Property `tensors` of `GGUFReader`. */
  public tensors: Record<
    string,
    { name: string; shape: bigint[]; type: GGUFTensorType; offset: bigint }
  > = {};
  /** Property `dataStart` of `GGUFReader`. */
  public dataStart: number = 0;
  /** Property `view` of `GGUFReader`. */
  private view: DataView;
  /** Property `offset` of `GGUFReader`. */
  private offset: number = 0;

  /**
   * Initializes a new `GGUFReader` instance.
   * @param buffer Parameter `buffer`.
   */
  constructor(public buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
    this.readHeader();
  }

  /**
   * Executes `readString` on `GGUFReader`.
   * @returns Result of `readString`.
   */
  private readString(): string {
    const len = Number(this.view.getBigUint64(this.offset, true));
    this.offset += 8;
    const decoder = new TextDecoder();
    const str = decoder.decode(new Uint8Array(this.buffer, this.offset, len));
    this.offset += len;
    return str;
  }

  /**
   * Executes `readVal` on `GGUFReader`.
   * @param vtype Parameter `vtype`.
   * @returns Result of `readVal`.
   */
  private readVal(vtype: GGUFValueType): ReturnType<typeof JSON.parse> {
    switch (vtype) {
      case GGUFValueType.UINT8: {
        const v = this.view.getUint8(this.offset);
        this.offset += 1;
        return v;
      }
      case GGUFValueType.INT8: {
        const v = this.view.getInt8(this.offset);
        this.offset += 1;
        return v;
      }
      case GGUFValueType.UINT16: {
        const v = this.view.getUint16(this.offset, true);
        this.offset += 2;
        return v;
      }
      case GGUFValueType.INT16: {
        const v = this.view.getInt16(this.offset, true);
        this.offset += 2;
        return v;
      }
      case GGUFValueType.UINT32: {
        const v = this.view.getUint32(this.offset, true);
        this.offset += 4;
        return v;
      }
      case GGUFValueType.INT32: {
        const v = this.view.getInt32(this.offset, true);
        this.offset += 4;
        return v;
      }
      case GGUFValueType.FLOAT32: {
        const v = this.view.getFloat32(this.offset, true);
        this.offset += 4;
        return v;
      }
      case GGUFValueType.UINT64: {
        const v = this.view.getBigUint64(this.offset, true);
        this.offset += 8;
        return v;
      }
      case GGUFValueType.INT64: {
        const v = this.view.getBigInt64(this.offset, true);
        this.offset += 8;
        return v;
      }
      case GGUFValueType.FLOAT64: {
        const v = this.view.getFloat64(this.offset, true);
        this.offset += 8;
        return v;
      }
      case GGUFValueType.BOOL: {
        const v = this.view.getUint8(this.offset) !== 0;
        this.offset += 1;
        return v;
      }
      case GGUFValueType.STRING:
        return this.readString();
      case GGUFValueType.ARRAY: {
        const atype = this.view.getUint32(this.offset, true);
        this.offset += 4;
        const len = Number(this.view.getBigUint64(this.offset, true));
        this.offset += 8;
        const arr = [];
        for (let i = 0; i < len; i++) {
          arr.push(this.readVal(atype));
        }
        return arr;
      }
      default:
        throw new Onnx9000Error(`Unknown value type: ${vtype}`);
    }
  }

  /**
   * Executes `readHeader` on `GGUFReader`.
   * @returns Result of `readHeader`.
   */
  private readHeader() {
    if (
      this.view.getUint8(0) !== 0x47 ||
      this.view.getUint8(1) !== 0x47 ||
      this.view.getUint8(2) !== 0x55 ||
      this.view.getUint8(3) !== 0x46
    ) {
      throw new Onnx9000Error('Not a GGUF file');
    }
    this.offset += 4;

    const version = this.view.getUint32(this.offset, true);
    this.offset += 4;
    if (version !== 2 && version !== 3) {
      throw new Onnx9000Error(`Unsupported GGUF version ${version}`);
    }

    const tensorCount = Number(this.view.getBigUint64(this.offset, true));
    this.offset += 8;
    const kvCount = Number(this.view.getBigUint64(this.offset, true));
    this.offset += 8;

    for (let i = 0; i < kvCount; i++) {
      const key = this.readString();
      const vtype = this.view.getUint32(this.offset, true);
      this.offset += 4;
      this.kvs[key] = this.readVal(vtype);
    }

    for (let i = 0; i < tensorCount; i++) {
      const name = this.readString();
      const nDims = this.view.getUint32(this.offset, true);
      this.offset += 4;
      const shape = [];
      for (let d = 0; d < nDims; d++) {
        shape.push(this.view.getBigUint64(this.offset, true));
        this.offset += 8;
      }
      const ttype = this.view.getUint32(this.offset, true);
      this.offset += 4;
      const offset = this.view.getBigUint64(this.offset, true);
      this.offset += 8;

      this.tensors[name] = { name, shape, type: ttype, offset };
    }

    const alignment = this.kvs['general.alignment'] || 32;
    const padding = (alignment - (this.offset % alignment)) % alignment;
    this.offset += padding;
    this.dataStart = this.offset;
  }

  /**
   * Executes `getTensor` on `GGUFReader`.
   * @param name Parameter `name`.
   * @returns Result of `getTensor`.
   */
  public getTensor(name: string): Uint8Array {
    if (!this.tensors[name]) throw new Onnx9000Error(`Tensor ${name} not found`);
    const t = this.tensors[name];

    let items = 1n;
    for (const d of t.shape) items *= d;

    let size = 0n;
    if (t.type === GGUFTensorType.F32) size = items * 4n;
    else if (t.type === GGUFTensorType.F16) size = items * 2n;
    else if (t.type === GGUFTensorType.Q4_0) size = (items / 32n) * 18n;
    else if (t.type === GGUFTensorType.Q4_1) size = (items / 32n) * 20n;
    else if (t.type === GGUFTensorType.Q8_0) size = (items / 32n) * 34n;
    else throw new Onnx9000Error('Unknown type');

    const byteOffset = this.dataStart + Number(t.offset);
    return new Uint8Array(this.buffer, byteOffset, Number(size));
  }
}
