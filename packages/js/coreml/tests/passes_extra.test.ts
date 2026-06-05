import { describe, it, expect } from 'vitest';
import { Builder, MILDataType, fuseAdjacentOps, Operation } from '../src/index.js';

describe('Passes Extras', () => {
  it('fuses adjacent split to concat', () => {
    const builder = new Builder();
    builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [4]));
    const s1 = builder.createVar('s1', builder.tensor(MILDataType.FLOAT32, [2]));
    const s2 = builder.createVar('s2', builder.tensor(MILDataType.FLOAT32, [2]));
    const y = builder.createVar('y', builder.tensor(MILDataType.FLOAT32, [4]));

    builder.addOp('split', { x }, [s1, s2]);
    const concatOp = builder.addOp('concat', { values: [s1, s2] }, [y]);

    fuseAdjacentOps(block);

    expect(block.operations[1]!.opType).toBe('identity');
    expect((block.operations[1]!.inputs['x'] as any).name).toBe('x');
  });

  it('fuses slice with adjacent pad', () => {
    const builder = new Builder();
    builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [4]));
    const p1 = builder.createVar('p1', builder.tensor(MILDataType.FLOAT32, [6]));
    const s1 = builder.createVar('s1', builder.tensor(MILDataType.FLOAT32, [4]));

    builder.addOp('pad', { x }, [p1], { pad_amounts: [1, 1] });
    const sliceOp = builder.addOp('slice_by_index', { x: p1 }, [s1]);

    fuseAdjacentOps(block);

    expect(block.operations[1]!.attributes['ane_hint_fused_pad']).toEqual([1, 1]);
  });

  it('covers CSE array inputs', async () => {
    const {
      deadCodeElimination,
      commonSubexpressionElimination,
      constantFolding,
      Program,
      Function,
      Block,
      Var,
      Operation,
      TensorType,
      MILDataType,
    } = await import('../src/index.js');

    const prog = new Program();
    const fn = new Function('main', [], []);
    const b = new Block('block0');
    b.operations = [];
    b.outputs = [];

    const v1 = new Var('v1', new TensorType(MILDataType.FLOAT32, [1]));
    const op1 = new Operation('const', { val: new Float32Array([1.0]) }, [v1]);
    op1.attributes['val'] = new Float32Array([1.0]);

    const v2 = new Var('v2', new TensorType(MILDataType.FLOAT32, [1]));
    const op2 = new Operation('const', { val: new Float32Array([1.0]) }, [v2]);
    op2.attributes['val'] = new Float32Array([1.0]);

    const v3 = new Var('v3', new TensorType(MILDataType.FLOAT32, [1]));
    const op3 = new Operation('concat', { inputs: [v1, v2] }, [v3]);

    const op4 = new Operation('add', { arr: [v1, v2], single: v1 }, [
      new Var('out_add', new TensorType(MILDataType.FLOAT32, [1])),
    ]);

    const unk = new Var('unkn', new TensorType(MILDataType.FLOAT32, [1]));
    const op5 = new Operation('add', { arr: [unk], single: unk }, [
      new Var('out_unk', new TensorType(MILDataType.FLOAT32, [1])),
    ]);
    b.inputs.push(unk);

    b.operations.push(op1, op2, op3, op4, op5);
    b.outputs.push(v3, op4.outputs[0], op5.outputs[0]);
    fn.blocks['block0'] = b;
    prog.functions['main'] = fn;

    deadCodeElimination(b);

    commonSubexpressionElimination(b);
    expect(op3.inputs['inputs']).toBeDefined();

    constantFolding(b);
  });
});
