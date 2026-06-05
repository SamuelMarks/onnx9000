/**
 * Zero-dependency FlatBuffer Reader.
 */
export class FlatBufferReader {
  private view: DataView;
  private bytes: Uint8Array;

  constructor(data: Uint8Array) {
    this.bytes = data;
    this.view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  }

  public getRoot(): number {
    const offset = this.view.getUint32(0, true);
    return offset;
  }

  public getRootAsModel(): number {
    // skip 4 bytes offset, skip identifier if present (which it is for TFL3)
    // Actually the root offset is just at byte 0.
    return this.view.getUint32(0, true);
  }

  public checkMagicBytes(magic: string): boolean {
    if (this.bytes.length < 8) return false;
    let actual = '';
    for (let i = 4; i < 8; i++) {
      actual += String.fromCharCode(this.bytes[i]!);
    }
    return actual === magic;
  }

  public getVTable(objOffset: number): number {
    return objOffset - this.view.getInt32(objOffset, true);
  }

  public getFieldOffset(objOffset: number, vtableIndex: number): number {
    const vtable = this.getVTable(objOffset);
    const vtableSize = this.view.getUint16(vtable, true);
    const fieldOffset = (vtableIndex + 2) * 2;

    if (fieldOffset >= vtableSize) return 0;

    const offsetInObject = this.view.getUint16(vtable + fieldOffset, true);
    if (offsetInObject === 0) return 0;

    return objOffset + offsetInObject;
  }

  public getInt8(objOffset: number, vtableIndex: number, def: number = 0): number {
    const offset = this.getFieldOffset(objOffset, vtableIndex);
    return offset !== 0 ? this.view.getInt8(offset) : def;
  }

  public getInt16(objOffset: number, vtableIndex: number, def: number = 0): number {
    const offset = this.getFieldOffset(objOffset, vtableIndex);
    return offset !== 0 ? this.view.getInt16(offset, true) : def;
  }

  public getInt32(objOffset: number, vtableIndex: number, def: number = 0): number {
    const offset = this.getFieldOffset(objOffset, vtableIndex);
    return offset !== 0 ? this.view.getInt32(offset, true) : def;
  }

  public getFloat32(objOffset: number, vtableIndex: number, def: number = 0.0): number {
    const offset = this.getFieldOffset(objOffset, vtableIndex);
    return offset !== 0 ? this.view.getFloat32(offset, true) : def;
  }

  public getVectorLength(vectorOffset: number): number {
    return this.view.getUint32(vectorOffset, true);
  }

  public getVectorItemOffset(vectorOffset: number, index: number, elemSize: number = 4): number {
    // 4 bytes length, followed by data
    return vectorOffset + 4 + index * elemSize;
  }

  public getString(objOffset: number, vtableIndex: number): string | null {
    let offset = this.getFieldOffset(objOffset, vtableIndex);
    if (offset === 0) return null;

    offset += this.view.getUint32(offset, true); // dereference string offset
    const length = this.view.getUint32(offset, true);

    let str = '';
    for (let i = 0; i < length; i++) {
      str += String.fromCharCode(this.bytes[offset + 4 + i]!);
    }
    return str;
  }

  public getIndirectOffset(objOffset: number, vtableIndex: number): number {
    const offset = this.getFieldOffset(objOffset, vtableIndex);
    if (offset === 0) return 0;
    return offset + this.view.getUint32(offset, true);
  }
}
