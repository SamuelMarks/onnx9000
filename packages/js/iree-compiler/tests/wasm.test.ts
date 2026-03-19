import { describe, it, expect } from 'vitest';
import { Type, Value, Region, Operation, Block } from '../src/ir/core.js';
import * as scf from '../src/dialects/web/scf.js';
import { lowerLinalgToSCF, WASMEmitter } from '../src/passes/lower_wasm.js';

describe('Web SCF Dialect & WASM Lowering', () => {
  it('should create scf ops', () => {
    const type = { id: 'index' };
    const val1 = new Value(type, {} as Operation);
    const region = new Region();

    expect(scf.forOp(val1, val1, val1, [], region).opcode).toBe('web.scf.for');
    expect(scf.yieldOp([]).opcode).toBe('web.scf.yield');
    expect(scf.ifOp(val1, region, region, []).opcode).toBe('web.scf.if');
    expect(scf.whileOp([], region, region, []).opcode).toBe('web.scf.while');
    expect(scf.condition(val1, []).opcode).toBe('web.scf.condition');
  });

  it('should lower generic to scf loops', () => {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    block.pushOperation(new Operation('web.linalg.generic', [], [], {}));
    lowerLinalgToSCF(region);

    const ops = region.blocks[0].operations;
    expect(ops.some((o) => o.opcode === 'web.scf.for')).toBe(true);
  });

  it('should emit WAT', () => {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    block.pushOperation(new Operation('web.scf.for', [], [], {}, []));
    block.pushOperation(new Operation('web.vm.add.i32', [], [], {}));
    block.pushOperation(new Operation('web.vm.add.v128', [], [], {}));

    const emitter = new WASMEmitter();
    const wat = emitter.emitWAT(region);

    expect(wat).toContain('(module');
    expect(wat).toContain('(loop $L1');
    expect(wat).toContain('i32.add');
    expect(wat).toContain('v128.add');
    expect(wat).toContain('(import "env" "memory"');

    const wasm = emitter.compileWATToWASM(wat);
    expect(wasm[0]).toBe(0x00);
    expect(wasm[1]).toBe(0x61);
  });
});
