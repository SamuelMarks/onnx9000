/** Protobuf Varint wire type. */
export const WIRE_TYPE_VARINT = 0;
/** Protobuf 64-bit wire type. */
export const WIRE_TYPE_64BIT = 1;
/** Protobuf Length-delimited wire type. */
export const WIRE_TYPE_LENGTH_DELIMITED = 2;
/** Protobuf 32-bit wire type. */
export const WIRE_TYPE_32BIT = 5;

/**
 * Utility for writing protobuf-encoded data to a growable buffer.
 */
export class BufferWriter {
  private buffer: Uint8Array;
  private offset: number = 0;

  /**
   * Create a new BufferWriter.
   * @param initialSize Starting buffer capacity in bytes.
   */
  constructor(initialSize: number = 1024) {
    this.buffer = new Uint8Array(initialSize);
  }

  /** Ensure the buffer has enough remaining capacity. */
  private ensureSpace(size: number) {
    if (this.offset + size > this.buffer.length) {
      const newBuffer = new Uint8Array(Math.max(this.buffer.length * 2, this.offset + size + 1024));
      newBuffer.set(this.buffer);
      this.buffer = newBuffer;
    }
  }

  /** Write a single byte. */
  writeByte(value: number) {
    this.ensureSpace(1);
    this.buffer[this.offset++] = value & 0xff;
  }

  /** Write a raw byte array. */
  writeBytes(bytes: Uint8Array) {
    this.ensureSpace(bytes.length);
    this.buffer.set(bytes, this.offset);
    this.offset += bytes.length;
  }

  /** Write a 32-bit VarInt. */
  writeVarInt(value: number) {
    if (value < 0) {
      this.writeVarInt64(value);
      return;
    }
    while (value > 127) {
      this.writeByte((value & 127) | 128);
      value >>>= 7;
    }
    this.writeByte(value & 127);
  }

  /** Write a 64-bit VarInt (handles BigInt). */
  writeVarInt64(value: bigint | number) {
    let bigVal = BigInt(value);
    if (bigVal < 0n) {
      // Convert to 64-bit unsigned representation
      bigVal = bigVal & 0xffffffffffffffffn;
    }
    while (bigVal > 127n) {
      this.writeByte(Number((bigVal & 127n) | 128n));
      bigVal >>= 7n;
    }
    this.writeByte(Number(bigVal & 127n));
  }

  /** Write a UTF-8 string with length prefix. */
  writeString(value: string) {
    const bytes = new TextEncoder().encode(value);
    this.writeVarInt(bytes.length);
    this.writeBytes(bytes);
  }

  /** Write a field tag. */
  writeTag(fieldNumber: number, wireType: number) {
    this.writeVarInt((fieldNumber << 3) | wireType);
  }

  /** Write a 32-bit float (little-endian). */
  writeFloat(value: number) {
    this.ensureSpace(4);
    const view = new DataView(this.buffer.buffer, this.buffer.byteOffset, this.buffer.byteLength);
    view.setFloat32(this.offset, value, true);
    this.offset += 4;
  }

  /** Write a 64-bit signed integer (little-endian). */
  writeInt64(value: bigint | number) {
    this.ensureSpace(8);
    const view = new DataView(this.buffer.buffer, this.buffer.byteOffset, this.buffer.byteLength);
    view.setBigInt64(this.offset, BigInt(value), true);
    this.offset += 8;
  }

  /** Finalize and return the written data as a Uint8Array. */
  getResult(): Uint8Array {
    return this.buffer.slice(0, this.offset);
  }
}
