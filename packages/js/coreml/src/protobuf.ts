/**
 * @fileoverview protobuf module for coreml.
 */
export class Writer {
  /** Property `chunks` of `Writer`. */
  private chunks: Uint8Array[] = [];
  /** Property `length` of `Writer`. */
  private length = 0;

  /**
   * Executes `writeBytes` on `Writer`.
   * @param bytes Parameter `bytes`.
   */
  writeBytes(bytes: Uint8Array): void {
    this.chunks.push(bytes);
    this.length += bytes.length;
  }

  /**
   * Executes `writeByte` on `Writer`.
   * @param byte Parameter `byte`.
   */
  writeByte(byte: number): void {
    this.writeBytes(new Uint8Array([byte]));
  }

  /**
   * Executes `writeVarInt` on `Writer`.
   * @param value Parameter `value`.
   */
  writeVarInt(value: number): void {
    let current = value;
    const chunk: number[] = [];
    while (current >= 0x80) {
      chunk.push((current & 0x7f) | 0x80);
      current >>>= 7;
    }
    chunk.push(current & 0x7f);
    this.writeBytes(new Uint8Array(chunk));
  }

  /**
   * Executes `writeVarInt64` on `Writer`.
   * @param value Parameter `value`.
   */
  writeVarInt64(value: bigint): void {
    let current = value;
    const chunk: number[] = [];
    while (current >= 0x80n) {
      chunk.push(Number(current & 0x7fn) | 0x80);
      current >>= 7n;
    }
    chunk.push(Number(current & 0x7fn));
    this.writeBytes(new Uint8Array(chunk));
  }

  /**
   * Executes `writeString` on `Writer`.
   * @param value Parameter `value`.
   */
  writeString(value: string): void {
    const bytes = new TextEncoder().encode(value);
    this.writeVarInt(bytes.length);
    this.writeBytes(bytes);
  }

  /**
   * Executes `writeTag` on `Writer`.
   * @param fieldNumber Parameter `fieldNumber`.
   * @param wireType Parameter `wireType`.
   */
  writeTag(fieldNumber: number, wireType: number): void {
    this.writeVarInt((fieldNumber << 3) | wireType);
  }

  /**
   * Executes `writeFloat` on `Writer`.
   * @param value Parameter `value`.
   */
  writeFloat(value: number): void {
    const buffer = new ArrayBuffer(4);
    new DataView(buffer).setFloat32(0, value, true); // true for little-endian
    this.writeBytes(new Uint8Array(buffer));
  }

  /**
   * Executes `writeDouble` on `Writer`.
   * @param value Parameter `value`.
   */
  writeDouble(value: number): void {
    const buffer = new ArrayBuffer(8);
    new DataView(buffer).setFloat64(0, value, true);
    this.writeBytes(new Uint8Array(buffer));
  }

  /**
   * Executes `writeFixed32` on `Writer`.
   * @param value Parameter `value`.
   */
  writeFixed32(value: number): void {
    const buffer = new ArrayBuffer(4);
    new DataView(buffer).setUint32(0, value, true);
    this.writeBytes(new Uint8Array(buffer));
  }

  /**
   * Executes `writeFixed64` on `Writer`.
   * @param value Parameter `value`.
   */
  writeFixed64(value: bigint): void {
    const buffer = new ArrayBuffer(8);
    new DataView(buffer).setBigUint64(0, value, true);
    this.writeBytes(new Uint8Array(buffer));
  }

  /**
   * Executes `finish` on `Writer`.
   * @returns Result of `finish`.
   */
  finish(): Uint8Array {
    const result = new Uint8Array(this.length);
    let offset = 0;
    for (const chunk of this.chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }
    return result;
  }
}
