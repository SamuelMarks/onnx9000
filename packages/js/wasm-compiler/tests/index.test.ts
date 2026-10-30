import { describe, expect, it } from 'vitest';
import { WasmCompiler } from '../src/index.js';

describe('WasmCompiler', () => {
  it('should encode unsigned LEB128 and compile empty and non-empty ONNX buffers to WASM', async () => {
    const compiler = new WasmCompiler();
    expect(Array.from(compiler.encodeLeb128U(0))).toEqual([0]);
    expect(Array.from(compiler.encodeLeb128U(624485))).toEqual([0xe5, 0x8e, 0x26]);
    expect(() => compiler.encodeLeb128U(-1)).toThrow(
      'encodeLeb128U requires a non-negative integer',
    );

    const modEmpty = await compiler.compile(new Uint8Array(0));
    expect(modEmpty).toBeInstanceOf(WebAssembly.Module);

    const modNonEmpty = await compiler.compile(new Uint8Array([1, 2, 3, 4]));
    expect(modNonEmpty).toBeInstanceOf(WebAssembly.Module);
  });
});
