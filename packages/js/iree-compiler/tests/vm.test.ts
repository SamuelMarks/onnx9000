import { describe, it, expect } from 'vitest';
import { Type, Value, Region, Operation, Block } from '../src/ir/core.js';
import * as vm from '../src/dialects/web/vm.js';
import {
  lowerHALToVM,
  BytecodeEmitter,
  disassembleWVM,
  optimizeAndAllocateRegisters,
} from '../src/passes/lower_vm.js';

describe('Web VM Dialect', () => {
  it('should create vm ops', () => {
    const type = { id: 'i32' };
    const val1 = new Value(type, {} as Operation);
    const region = new Region();
    const block = new Block(region);

    expect(vm.moduleOp(region).opcode).toBe('web.vm.module');
    expect(vm.func('main', [], [], region).opcode).toBe('web.vm.func');
    expect(vm.call('func1', [val1], [type]).opcode).toBe('web.vm.call');
    expect(vm.branch(block, [val1]).opcode).toBe('web.vm.branch');
    expect(vm.condBranch(val1, block, [], block, []).opcode).toBe('web.vm.cond_branch');
    expect(vm.cmp('eq', val1, val1, type).opcode).toBe('web.vm.cmp');
    expect(vm.addI32(val1, val1, type).opcode).toBe('web.vm.add.i32');
    expect(vm.mulI32(val1, val1, type).opcode).toBe('web.vm.mul.i32');
    expect(vm.returnOp([val1]).opcode).toBe('web.vm.return');
    expect(vm.importOp('log', 'console', 'log').opcode).toBe('web.vm.import');
  });

  it('should lower HAL to VM', () => {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    block.pushOperation(
      new Operation('web.hal.command_buffer.create', [], [], { mode: 'one_shot' }),
    );

    lowerHALToVM(region);

    expect(region.blocks.length).toBe(1);
    const newOp = region.blocks[0].operations[0];
    expect(newOp.opcode).toBe('web.vm.module');

    const emitter = new BytecodeEmitter();
    const bytecode = emitter.emit(region);
    expect(bytecode[0]).toBe(0x57); // W
    expect(bytecode[1]).toBe(0x56); // V
    expect(bytecode[2]).toBe(0x4d); // M
    expect(bytecode[3]).toBe(0x30); // 0

    const asm = disassembleWVM(bytecode);
    expect(asm).toContain('WVM0 Header OK');
    expect(asm).toContain('Module');
  });
});

it('covers disassemble cases', () => {
  const buf = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x01, 0x02, 0x03, 0xff]);
  const asm = disassembleWVM(buf);
  expect(asm).toContain('Module');
  expect(asm).toContain('Func');
  expect(asm).toContain('Call');
  expect(asm).toContain('Unknown(0xff)');
});

it('should hit emit bytecode cases', () => {
  const region = new Region();
  const block = new Block(region);
  region.pushBlock(block);
  block.pushOperation(new Operation('web.vm.func', [], []));
  block.pushOperation(new Operation('web.vm.call', [], []));
  block.pushOperation(new Operation('unknown.op', [], []));

  const emitter = new BytecodeEmitter();
  const bytecode = emitter.emit(region);
  expect(bytecode.length).toBeGreaterThan(4);
});

it('should optimize and allocate registers', () => {
  expect(() => optimizeAndAllocateRegisters(new Region())).not.toThrow();
});

it('should lower other hal ops', () => {
  const region = new Region();
  const block = new Block(region);
  region.pushBlock(block);

  block.pushOperation(new Operation('web.hal.command_buffer.dispatch', [], []));
  block.pushOperation(new Operation('web.hal.buffer.subspan', [], []));

  lowerHALToVM(region);
  expect(region.blocks.length).toBe(1);
});
