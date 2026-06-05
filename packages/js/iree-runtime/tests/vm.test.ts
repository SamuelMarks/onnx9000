import { describe, it, expect } from 'vitest';
import { Module, Context, WVMInterpreter, HALBindings, WASMWVMInterpreter } from '../src/vm.js';

describe('iree-runtime vm', () => {
  it('should run wvm interpreter', async () => {
    const mod = new Module();
    const ctx = new Context(mod);
    HALBindings.register(ctx, {});

    const bc = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x03, 0xff]);
    const interp = new WVMInterpreter(bc, ctx);

    interp.runSync();
    await interp.runAsync();

    expect(ctx.pc).toBeGreaterThan(0);
  });

  it('should throw on invalid bc', () => {
    const mod = new Module();
    const ctx = new Context(mod);
    expect(() => new WVMInterpreter(new Uint8Array([0, 0, 0, 0]), ctx)).toThrow();
  });
});
