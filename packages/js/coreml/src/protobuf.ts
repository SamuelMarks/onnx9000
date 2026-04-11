/* eslint-disable */
import {
  readVarInt,
  Reader,
  WIRE_TYPE_VARINT,
  WIRE_TYPE_64BIT,
  WIRE_TYPE_LENGTH_DELIMITED,
  WIRE_TYPE_32BIT,
} from '@onnx9000/core';

export class Writer {
  private chunks: Uint8Array[] = [];
  private length = 0;

  writeBytes(bytes: Uint8Array): void {
    this.chunks.push(bytes);
    this.length += bytes.length;
  }

  writeByte(byte: number): void {
    this.writeBytes(new Uint8Array([byte]));
  }

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

  writeString(value: string): void {
    const bytes = new TextEncoder().encode(value);
    this.writeVarInt(bytes.length);
    this.writeBytes(bytes);
  }

  writeTag(fieldNumber: number, wireType: number): void {
    this.writeVarInt((fieldNumber << 3) | wireType);
  }

  writeFloat(value: number): void {
    const buffer = new ArrayBuffer(4);
    new DataView(buffer).setFloat32(0, value, true); // true for little-endian
    this.writeBytes(new Uint8Array(buffer));
  }

  writeDouble(value: number): void {
    const buffer = new ArrayBuffer(8);
    new DataView(buffer).setFloat64(0, value, true);
    this.writeBytes(new Uint8Array(buffer));
  }

  writeFixed32(value: number): void {
    const buffer = new ArrayBuffer(4);
    new DataView(buffer).setUint32(0, value, true);
    this.writeBytes(new Uint8Array(buffer));
  }

  writeFixed64(value: bigint): void {
    const buffer = new ArrayBuffer(8);
    new DataView(buffer).setBigUint64(0, value, true);
    this.writeBytes(new Uint8Array(buffer));
  }

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
