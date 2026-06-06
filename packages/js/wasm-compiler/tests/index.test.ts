import { describe, expect, it, vi } from 'vitest';
import { WasmCompiler } from '../src/index.js';

describe('WasmCompiler', () => {
  it('should compile', async () => {
    const c = new WasmCompiler();

    global.WebAssembly = {
      compile: vi.fn().mockResolvedValue({}),
    } as any;

    const m = await c.compile(new Uint8Array());
    expect(m).toBeDefined();
  });
});
