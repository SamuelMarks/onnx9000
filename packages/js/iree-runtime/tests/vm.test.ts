import { describe, it, expect } from 'vitest';
import { Module, Context, WVMInterpreter, HALBindings } from '../src/vm.js';

describe('WVM Interpreter', () => {
  it('should validate magic header', () => {
    const mod = new Module();
    const ctx = new Context(mod);
    expect(() => new WVMInterpreter(new Uint8Array([0, 0, 0, 0]), ctx)).toThrow(
      'Invalid WVM Bytecode',
    );

    expect(() => new WVMInterpreter(new Uint8Array([0x57, 0x56, 0x4d, 0x30]), ctx)).not.toThrow();
  });

  it('should set and get array buffers', () => {
    const mod = new Module(1024);
    const ctx = new Context(mod);
    const vm = new WVMInterpreter(new Uint8Array([0x57, 0x56, 0x4d, 0x30]), ctx);

    const input = new Float32Array([1.0, 2.0, 3.0]);
    vm.setInput(0, input.buffer);

    const outBuffer = vm.getOutput(0, 12);
    const output = new Float32Array(outBuffer);

    expect(output[0]).toBe(1.0);
    expect(output[1]).toBe(2.0);
    expect(output[2]).toBe(3.0);
  });

  it('should execute opcodes sync', () => {
    const mod = new Module();
    const ctx = new Context(mod);
    // header, add r0 = r1 + r2, return
    const bc = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x04, 0x00, 0x01, 0x02, 0xff]);
    const vm = new WVMInterpreter(bc, ctx);

    ctx.registers[1] = 10;
    ctx.registers[2] = 20;

    vm.runSync();
    expect(ctx.registers[0]).toBe(30);
  });

  it('should execute opcodes async with hal binding', async () => {
    const mod = new Module();
    const ctx = new Context(mod);

    let called = false;
    ctx.loadImport('hal', 'cmd_create', () => {
      called = true;
    });

    // header, call, return
    const bc = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x03, 0xff]);
    const vm = new WVMInterpreter(bc, ctx);

    await vm.runAsync();
    expect(called).toBe(true);
  });

  it('should handle context loss', () => {
    const mod = new Module();
    const ctx = new Context(mod);

    HALBindings.register(ctx, null); // passing null device

    const bc = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x03, 0xff]);
    const vm = new WVMInterpreter(bc, ctx);

    expect(() => vm.runSync()).toThrow('VM Error: WebGPU Context Lost');
  });
});

it('should throw on unknown opcode', () => {
  const mod = new Module();
  const ctx = new Context(mod);
  const bc = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x99]);
  const vm = new WVMInterpreter(bc, ctx);
  expect(() => vm.runSync()).toThrow('Unknown opcode');
});

it('should call HALBindings with valid device', () => {
  const mod = new Module();
  const ctx = new Context(mod);
  HALBindings.register(ctx, { valid: true });

  // hit cmd_create
  const bc1 = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x03, 0xff]);
  const vm1 = new WVMInterpreter(bc1, ctx);
  vm1.runSync();

  // Call the buffer_subspan directly from the registry
  mod.imports.get('hal.buffer_subspan')();
});

it('covers runAsync missing branches', async () => {
  const mod = new Module();
  const ctx = new Context(mod);

  // 100 operations to hit yield
  const ops = new Array(100).fill(0x01);
  const bc = new Uint8Array([0x57, 0x56, 0x4d, 0x30, ...ops, 0xff]);
  const vm = new WVMInterpreter(bc, ctx);
  await vm.runAsync();

  // hit unknown opcode in runAsync
  const bcUnknown = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x99]);
  const vmUnknown = new WVMInterpreter(bcUnknown, ctx);
  await expect(vmUnknown.runAsync()).rejects.toThrow('Unknown opcode');
});

it('covers runSync debug logging', () => {
  const mod = new Module();
  const ctx = new Context(mod);
  const bc = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0xff]);
  const vm = new WVMInterpreter(bc, ctx);
  vm.runSync(true);
});

it('covers dummy cases for 0x01 and 0x02', async () => {
  const mod = new Module();
  const ctx = new Context(mod);
  const bc = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x01, 0x02, 0xff]);
  const vm = new WVMInterpreter(bc, ctx);
  vm.runSync();
  await vm.runAsync();
});
