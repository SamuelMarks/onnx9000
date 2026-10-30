/**
 * @fileoverview GGUF binary header parser and writer utilities for `@onnx9000/gguf`.
 * Supports inspecting GGUF magic headers, version numbers, tensor counts, and KV counts.
 */

/**
 * Parsed GGUF binary header summary.
 */
import { Onnx9000Error } from '@onnx9000/core';
/** Interface `GgufHeader` definition. */
export interface GgufHeader {
  /** 4-character ASCII magic string ('GGUF'). */
  magic: string;
  /** GGUF format specification version (v2 or v3). */
  version: number;
  /** Number of tensors stored in the GGUF file. */
  tensorCount: bigint;
  /** Number of metadata key-value entries in the GGUF header. */
  kvCount: bigint;
}

/**
 * GGUF header serializer and parser.
 */
export class Gguf {
  /** Standard GGUF magic bytes ('GGUF'). */
  public static readonly MAGIC = 0x46554747;

  /**
   * Serializes a minimal 24-byte GGUF v3 header into a `Uint8Array`.
   * @param tensorCount Number of tensors in the file.
   * @param kvCount Number of metadata key-value pairs.
   * @param version GGUF format version (defaults to 3).
   * @returns A 24-byte `Uint8Array` containing the serialized GGUF header.
   */
  public writeHeader(tensorCount = 0n, kvCount = 0n, version = 3): Uint8Array {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setUint32(0, Gguf.MAGIC, true);
    view.setUint32(4, version, true);
    view.setBigUint64(8, tensorCount, true);
    view.setBigUint64(16, kvCount, true);
    return new Uint8Array(buf);
  }

  /**
   * Parses a 24-byte GGUF binary header from a `Uint8Array`.
   * @param bytes Input binary buffer (at least 24 bytes).
   * @returns Parsed `GgufHeader` metadata.
   * @throws Error if the buffer is too short or does not start with 'GGUF'.
   */
  public readHeader(bytes: Uint8Array): GgufHeader {
    if (bytes.byteLength < 24) {
      throw new Onnx9000Error('GGUF buffer must be at least 24 bytes');
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magicNum = view.getUint32(0, true);
    if (magicNum !== Gguf.MAGIC) {
      throw new Onnx9000Error('Invalid GGUF magic header');
    }
    const version = view.getUint32(4, true);
    const tensorCount = view.getBigUint64(8, true);
    const kvCount = view.getBigUint64(16, true);
    return { magic: 'GGUF', version, tensorCount, kvCount };
  }

  /**
   * Runs a self-check roundtrip of GGUF header serialization and parsing.
   * @returns Status string confirming GGUF processing.
   */
  public run(): string {
    const bytes = this.writeHeader(2n, 4n, 3);
    this.readHeader(bytes);
    return '[gguf] processed';
  }
}
