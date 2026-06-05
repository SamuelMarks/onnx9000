import { describe, it, expect } from 'vitest';
import {
  Builder,
  MILDataType,
  replaceOperation,
  replaceVarUsage,
  inferShapes,
  Operation,
  TensorType,
} from '../src/index.js';

describe('MIL Rewriter', () => {
  it('replaces operations successfully', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [1]));
    const out1 = builder.createVar('out1', builder.tensor(MILDataType.FLOAT32, [1]));

    const op1 = builder.addOp('const', {}, [x]);
    const op2 = builder.addOp('relu', { x }, [out1]);

    expect(block.operations.length).toBe(2);

    const out2 = builder.createVar('out2', builder.tensor(MILDataType.FLOAT32, [1]));
    const newOp1 = new Operation('sigmoid', { x }, [out2]);
    const newOp2 = new Operation('tanh', { x: out2 }, [out1]);

    replaceOperation(block, op2, [newOp1, newOp2]);

    expect(block.operations.length).toBe(3);
    expect(block.operations[1]!.opType).toBe('sigmoid');
    expect(block.operations[2]!.opType).toBe('tanh');

    // test not found
    const fakeOp = new Operation('fake', {}, []);
    expect(() => replaceOperation(block, fakeOp, [])).toThrow('not found in block');
  });

  it('replaces var usage across the block', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [1]));
    const oldY = builder.createVar('y', builder.tensor(MILDataType.FLOAT32, [1]));
    const newY = builder.createVar('new_y', builder.tensor(MILDataType.FLOAT32, [1]));
    const z = builder.createVar('z', builder.tensor(MILDataType.FLOAT32, [1]));

    const op1 = builder.addOp('add', { x, y: oldY }, [z]);
    // also as array input
    const op2 = builder.addOp('concat', { vals: [x, oldY] }, [oldY]);

    block.outputs = [oldY];

    replaceVarUsage(block, oldY, newY);

    expect((op1.inputs['y'] as any).name).toBe('new_y');
    expect((op2.inputs['vals'] as any)[1].name).toBe('new_y');
    expect(block.outputs[0]!.name).toBe('new_y');
  });

  it('infers shapes for basic operations', () => {
    const builder = new Builder();
    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    // Create vars with shapes
    const x = builder.createVar('x', builder.tensor(MILDataType.FLOAT32, [1, 3, 224, 224]));
    block.inputs.push(x);

    const y = builder.createVar('y', builder.tensor(MILDataType.FLOAT32, [1]));
    builder.addOp('const', {}, [y]);

    // Create unknown out
    const z = builder.createVar('z', builder.tensor(MILDataType.FLOAT32, []));
    builder.addOp('add', { x, y }, [z]);

    // Unknown passthrough
    const out = builder.createVar('out', builder.tensor(MILDataType.FLOAT32, []));
    builder.addOp('relu', { x: z }, [out]);

    // Multiple inputs but unknown passthrough
    const out2 = builder.createVar('out2', builder.tensor(MILDataType.FLOAT32, []));
    builder.addOp('fake_op', { a: z, b: y }, [out2]); // a is z so it grabs z's shape first

    inferShapes(block);

    expect((z.type as TensorType).shape).toEqual([1, 3, 224, 224]);
    expect((out.type as TensorType).shape).toEqual([1, 3, 224, 224]);
    expect((out2.type as TensorType).shape).toEqual([1, 3, 224, 224]); // it takes z's shape as fallback
  });
});
