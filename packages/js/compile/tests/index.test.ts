import { describe, expect, it } from 'vitest';
import { Compile } from '../src/index.js';

describe('Compile', () => {
  it('should run default compilation pipeline', () => {
    const c = new Compile();
    expect(c.run()).toBe('[compile] processed');
  });

  it('should compile ops to C, WGSL, and WASM targets', () => {
    const c = new Compile({ target: 'c', optLevel: 3, prefix: 'net_' });
    const cArt = c.compileOps(['Conv', 'Relu']);
    expect(cArt.target).toBe('c');
    expect(cArt.optLevel).toBe(3);
    expect(cArt.code).toContain('void net_forward(void)');

    const wgslArt = c.compileOps(['MatMul'], { target: 'wgsl', prefix: 'gpu_' });
    expect(wgslArt.target).toBe('wgsl');
    expect(wgslArt.code).toContain('@compute @workgroup_size(64)');

    const wasmArt = c.compileOps(['Add'], { target: 'wasm' });
    expect(wasmArt.target).toBe('wasm');
    expect(wasmArt.binary.slice(0, 4)).toEqual(new Uint8Array([0x00, 0x61, 0x73, 0x6d]));
  });
});
