/**
 * @fileoverview reader module for tflite-exporter.
 */
/**
 * Zero-dependency FlatBuffer Reader.
 */
export class FlatBufferReader {
  /** Property `view` of `FlatBufferReader`. */
  private view: DataView;
  /** Property `bytes` of `FlatBufferReader`. */
  private bytes: Uint8Array;

  /**
   * Initializes a new `FlatBufferReader` instance.
   * @param data Parameter `data`.
   */
  constructor(data: Uint8Array) {
    this.bytes = data;
    this.view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  }

  /**
   * Executes `getRoot` on `FlatBufferReader`.
   * @returns Result of `getRoot`.
   */
  public getRoot(): number {
    const offset = this.view.getUint32(0, true);
    return offset;
  }

  /**
   * Executes `getRootAsModel` on `FlatBufferReader`.
   * @returns Result of `getRootAsModel`.
   */
  public getRootAsModel(): number {
    // skip 4 bytes offset, skip identifier if present (which it is for TFL3)
    // Actually the root offset is just at byte 0.
    return this.view.getUint32(0, true);
  }

  /**
   * Executes `checkMagicBytes` on `FlatBufferReader`.
   * @param magic Parameter `magic`.
   * @returns Result of `checkMagicBytes`.
   */
  public checkMagicBytes(magic: string): boolean {
    if (this.bytes.length < 8) return false;
    let actual = '';
    for (let i = 4; i < 8; i++) {
      actual += String.fromCharCode(this.bytes[i] as number);
    }
    return actual === magic;
  }

  /**
   * Executes `getVTable` on `FlatBufferReader`.
   * @param objOffset Parameter `objOffset`.
   * @returns Result of `getVTable`.
   */
  public getVTable(objOffset: number): number {
    return objOffset - this.view.getInt32(objOffset, true);
  }

  /**
   * Executes `getFieldOffset` on `FlatBufferReader`.
   * @param objOffset Parameter `objOffset`.
   * @param vtableIndex Parameter `vtableIndex`.
   * @returns Result of `getFieldOffset`.
   */
  public getFieldOffset(objOffset: number, vtableIndex: number): number {
    const vtable = this.getVTable(objOffset);
    const vtableSize = this.view.getUint16(vtable, true);
    const fieldOffset = (vtableIndex + 2) * 2;

    if (fieldOffset >= vtableSize) return 0;

    const offsetInObject = this.view.getUint16(vtable + fieldOffset, true);
    if (offsetInObject === 0) return 0;

    return objOffset + offsetInObject;
  }

  /**
   * Executes `getInt8` on `FlatBufferReader`.
   * @param objOffset Parameter `objOffset`.
   * @param vtableIndex Parameter `vtableIndex`.
   * @param def Parameter `def`.
   * @returns Result of `getInt8`.
   */
  public getInt8(objOffset: number, vtableIndex: number, def: number = 0): number {
    const offset = this.getFieldOffset(objOffset, vtableIndex);
    return offset !== 0 ? this.view.getInt8(offset) : def;
  }

  /**
   * Executes `getInt16` on `FlatBufferReader`.
   * @param objOffset Parameter `objOffset`.
   * @param vtableIndex Parameter `vtableIndex`.
   * @param def Parameter `def`.
   * @returns Result of `getInt16`.
   */
  public getInt16(objOffset: number, vtableIndex: number, def: number = 0): number {
    const offset = this.getFieldOffset(objOffset, vtableIndex);
    return offset !== 0 ? this.view.getInt16(offset, true) : def;
  }

  /**
   * Executes `getInt32` on `FlatBufferReader`.
   * @param objOffset Parameter `objOffset`.
   * @param vtableIndex Parameter `vtableIndex`.
   * @param def Parameter `def`.
   * @returns Result of `getInt32`.
   */
  public getInt32(objOffset: number, vtableIndex: number, def: number = 0): number {
    const offset = this.getFieldOffset(objOffset, vtableIndex);
    return offset !== 0 ? this.view.getInt32(offset, true) : def;
  }

  /**
   * Executes `getFloat32` on `FlatBufferReader`.
   * @param objOffset Parameter `objOffset`.
   * @param vtableIndex Parameter `vtableIndex`.
   * @param def Parameter `def`.
   * @returns Result of `getFloat32`.
   */
  public getFloat32(objOffset: number, vtableIndex: number, def: number = 0.0): number {
    const offset = this.getFieldOffset(objOffset, vtableIndex);
    return offset !== 0 ? this.view.getFloat32(offset, true) : def;
  }

  /**
   * Executes `getVectorLength` on `FlatBufferReader`.
   * @param vectorOffset Parameter `vectorOffset`.
   * @returns Result of `getVectorLength`.
   */
  public getVectorLength(vectorOffset: number): number {
    return this.view.getUint32(vectorOffset, true);
  }

  /**
   * Executes `getVectorItemOffset` on `FlatBufferReader`.
   * @param vectorOffset Parameter `vectorOffset`.
   * @param index Parameter `index`.
   * @param elemSize Parameter `elemSize`.
   * @returns Result of `getVectorItemOffset`.
   */
  public getVectorItemOffset(vectorOffset: number, index: number, elemSize: number = 4): number {
    // 4 bytes length, followed by data
    return vectorOffset + 4 + index * elemSize;
  }

  /**
   * Executes `getString` on `FlatBufferReader`.
   * @param objOffset Parameter `objOffset`.
   * @param vtableIndex Parameter `vtableIndex`.
   * @returns Result of `getString`.
   */
  public getString(objOffset: number, vtableIndex: number): string | null {
    let offset = this.getFieldOffset(objOffset, vtableIndex);
    if (offset === 0) return null;

    offset += this.view.getUint32(offset, true); // dereference string offset
    const length = this.view.getUint32(offset, true);

    let str = '';
    for (let i = 0; i < length; i++) {
      str += String.fromCharCode(this.bytes[offset + 4 + i] as number);
    }
    return str;
  }

  /**
   * Executes `getIndirectOffset` on `FlatBufferReader`.
   * @param objOffset Parameter `objOffset`.
   * @param vtableIndex Parameter `vtableIndex`.
   * @returns Result of `getIndirectOffset`.
   */
  public getIndirectOffset(objOffset: number, vtableIndex: number): number {
    const offset = this.getFieldOffset(objOffset, vtableIndex);
    if (offset === 0) return 0;
    return offset + this.view.getUint32(offset, true);
  }
}
