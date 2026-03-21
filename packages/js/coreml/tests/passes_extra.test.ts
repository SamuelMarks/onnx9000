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
});
