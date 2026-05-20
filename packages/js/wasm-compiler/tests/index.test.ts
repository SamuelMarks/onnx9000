import { describe, it, expect } from 'vitest';
import { WasmCompiler } from '../src/index.js';

describe('WasmCompiler', () => {
  it('compiles valid buffer to WebAssembly.Module', async () => {
    const compiler = new WasmCompiler();
    const result = await compiler.compile(new Uint8Array([1, 2, 3]));
    expect(result).toBeInstanceOf(WebAssembly.Module);
  });
});
