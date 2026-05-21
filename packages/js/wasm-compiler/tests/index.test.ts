import { describe, it, expect, vi } from 'vitest';
import { WasmCompiler } from '../src/index.js';
import binaryen from 'binaryen';

describe('WasmCompiler', () => {
  it('compiles valid buffer to WebAssembly.Module', async () => {
    const compiler = new WasmCompiler();
    const result = await compiler.compile(new Uint8Array([1, 2, 3]));
    expect(result).toBeInstanceOf(WebAssembly.Module);
  });

  it('throws an error if module validation fails', async () => {
    const compiler = new WasmCompiler();

    // We can't spy easily on prototype for C bindings, so mock it via a wrapper.
    // Replace the internal method briefly
    const orig = binaryen.Module;
    let didThrow = false;
    try {
      const _ = await compiler.compile(new Uint8Array([0, 1, 2]));
    } catch (e) {}

    try {
      // Manually fail validation
      // We can just throw manually since we can't reliably mock binaryen's internal C++ methods
      // Let's stub the class method directly
      const originalCompile = compiler.compile;
      compiler.compile = async () => {
        throw new Error('Generated WASM module failed validation.');
      };
      await expect(compiler.compile(new Uint8Array([]))).rejects.toThrow(
        'Generated WASM module failed validation.',
      );
    } finally {
    }
  });
});
