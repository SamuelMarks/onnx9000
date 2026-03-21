import { describe, it, expect } from 'vitest';
import { MLModelCCompiler } from '../src/index.js';

describe('MLModelCCompiler (WASM Stub)', () => {
  it('Returns a Uint8Array stub when compiling', () => {
    // 293. Build a WASM fallback for reading/writing Apple's compiled .mlmodelc binary directories directly.
    const input = new Uint8Array([1, 2, 3]);
    const output = MLModelCCompiler.compile(input);
    expect(output).toBeInstanceOf(Uint8Array);
    expect(output.length).toBe(0); // Currently a stub
  });
});
