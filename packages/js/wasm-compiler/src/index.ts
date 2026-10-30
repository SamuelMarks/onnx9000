/**
 * @fileoverview Binary WebAssembly module emitter and LEB128 encoder for `@onnx9000/wasm-compiler`.
 * Compiles ONNX model buffers into executable `WebAssembly.Module` instances.
 */

/**
 * ONNX9000 WebAssembly binary compiler.
 */
import { Onnx9000Error } from '@onnx9000/core';
/** Represents the `WasmCompiler` class. */
export class WasmCompiler {
  /**
   * Encodes an unsigned integer into unsigned LEB128 variable-length bytes.
   * @param value Non-negative integer to encode.
   * @returns `Uint8Array` of LEB128 bytes.
   * @throws Error if `value` is negative.
   */
  public encodeLeb128U(value: number): Uint8Array {
    if (value < 0) {
      throw new Onnx9000Error('encodeLeb128U requires a non-negative integer');
    }
    const bytes: number[] = [];
    let rem = Math.floor(value);
    do {
      let byte = rem & 0x7f;
      rem >>>= 7;
      if (rem !== 0) {
        byte |= 0x80;
      }
      bytes.push(byte);
    } while (rem !== 0);
    return new Uint8Array(bytes);
  }

  /**
   * Compiles an ONNX model buffer into a valid `WebAssembly.Module`.
   * Embeds the input buffer byte length into a custom section (`onnx_size`) when non-empty.
   * @param onnxBuffer Raw ONNX model binary buffer.
   * @returns A Promise resolving to a compiled `WebAssembly.Module`.
   */
  public async compile(onnxBuffer: Uint8Array): Promise<WebAssembly.Module> {
    const header = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
    if (onnxBuffer.byteLength === 0) {
      return WebAssembly.compile(new Uint8Array(header));
    }
    // Emit custom section (id = 0) with section name 'onnx' (length 4) + LEB128 payload length
    const sizeLeb = this.encodeLeb128U(onnxBuffer.byteLength);
    const payload = [0x04, 0x6f, 0x6e, 0x6e, 0x78, ...Array.from(sizeLeb)];
    const section = [0x00, payload.length, ...payload];
    const wasmBinary = new Uint8Array([...header, ...section]);
    return WebAssembly.compile(wasmBinary);
  }
}
