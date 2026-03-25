export const WIRE_TYPE_VARINT = 0;
export const WIRE_TYPE_64BIT = 1;
export const WIRE_TYPE_LENGTH_DELIMITED = 2;
export const WIRE_TYPE_32BIT = 5;

export class BufferWriter {
  private buffer: Uint8Array;
  private offset: number = 0;

  constructor(initialSize: number = 1024) {
    this.buffer = new Uint8Array(initialSize);
  }

  private ensureSpace(size: number) {
    if (this.offset + size > this.buffer.length) {
      const newBuffer = new Uint8Array(Math.max(this.buffer.length * 2, this.offset + size + 1024));
      newBuffer.set(this.buffer);
      this.buffer = newBuffer;
    }
  }

  writeByte(value: number) {
    this.ensureSpace(1);
    this.buffer[this.offset++] = value & 0xff;
  }

  writeBytes(bytes: Uint8Array) {
    this.ensureSpace(bytes.length);
    this.buffer.set(bytes, this.offset);
    this.offset += bytes.length;
  }

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

  writeString(value: string) {
    const bytes = new TextEncoder().encode(value);
    this.writeVarInt(bytes.length);
    this.writeBytes(bytes);
  }

  writeTag(fieldNumber: number, wireType: number) {
    this.writeVarInt((fieldNumber << 3) | wireType);
  }

  writeFloat(value: number) {
    this.ensureSpace(4);
    const view = new DataView(this.buffer.buffer, this.buffer.byteOffset, this.buffer.byteLength);
    view.setFloat32(this.offset, value, true);
    this.offset += 4;
  }

  writeInt64(value: bigint | number) {
    this.ensureSpace(8);
    const view = new DataView(this.buffer.buffer, this.buffer.byteOffset, this.buffer.byteLength);
    view.setBigInt64(this.offset, BigInt(value), true);
    this.offset += 8;
  }

  getResult(): Uint8Array {
    return this.buffer.slice(0, this.offset);
  }
}
