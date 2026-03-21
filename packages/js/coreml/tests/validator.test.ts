import { describe, it, expect } from 'vitest';
import { Program, Function, Block, validateMILProgram, validateBlock } from '../src/index.js';
import { Builder, MILDataType } from '../src/index.js';

describe('MIL Block Validators', () => {
  it('Validates perfectly formed functional graphs', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [1]));
    const out = builder.createVar('out', builder.tensor(MILDataType.FLOAT32, [1]));

    builder.addOp('const', {}, [x]);
    builder.addOp('relu', { x }, [out]);
    block.outputs = [out];

    expect(() => validateBlock(block)).not.toThrow();
  });

  it('Throws an error when consumed variables are not produced', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const unbound = builder.createVar('missing_input', builder.tensor(MILDataType.FLOAT32, [1]));
    const out = builder.createVar('out', builder.tensor(MILDataType.FLOAT32, [1]));

    // We add relu consuming a variable never produced
    builder.addOp('relu', { x: unbound }, [out]);
    block.outputs = [out];

    expect(() => validateBlock(block)).toThrowError(
      /Operation input missing_input is not available/,
    );
  });

  it('Validates complete MILProgram containers', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');
    const prog = builder.createProgram();

    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [1]));
    builder.addOp('const', {}, [x]);
    block.outputs = [x];

    expect(validateMILProgram(prog)).toBe(true);
  });

  it('Throws DAG error on circular topologies', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const a = builder.createVar('a', builder.tensor(MILDataType.FLOAT32, [1]));
    const b = builder.createVar('b', builder.tensor(MILDataType.FLOAT32, [1]));

    // Cycle A->B->A
    builder.addOp('add', { x: b, y: b }, [a]);
    builder.addOp('sub', { x: a, y: a }, [b]);

    expect(() => validateBlock(block)).toThrowError(/not a valid DAG/);
  });
});
