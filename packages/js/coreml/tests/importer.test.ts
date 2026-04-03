import { describe, it, expect } from 'vitest';
import { MILToONNXConverter } from '../src/importer.js';
import { Program, Function, Block, Operation, Var } from '../src/mil/ast.js';
import { TensorType, MILDataType } from '../src/mil/types.js';

describe('MILToONNXConverter', () => {
  it('converts a simple program', () => {
    const prog = new Program();
    const v = new Var('in1', new TensorType(MILDataType.FLOAT32, [1]));
    const out_v = new Var('out_v', new TensorType(MILDataType.FLOAT32, [1]));
    const fn = new Function('main', [v], [out_v]);

    const b = new Block('block0');
    const op1 = new Operation(
      'constexpr_affine_dequantize',
      {
        quantized_data: new Var('qd', new TensorType(MILDataType.INT8, [1])),
        zero_point: new Var('zp', new TensorType(MILDataType.INT8, [1])),
        scale: new Var('sc', new TensorType(MILDataType.FLOAT32, [1])),
      },
      [],
    );
    op1.outputs = [new Var('out1', new TensorType(MILDataType.FLOAT32, [1]))];
    b.operations.push(op1);

    const op2 = new Operation('constexpr_lut_dequantize', {}, []);
    op2.outputs = [new Var('out2', new TensorType(MILDataType.FLOAT32, [1]))];
    b.operations.push(op2);

    const op3 = new Operation(
      'add',
      {
        x: new Var('in1', new TensorType(MILDataType.FLOAT32, [1])),
        y: new Var('out1', new TensorType(MILDataType.FLOAT32, [1])),
      },
      [],
    );
    op3.outputs = [new Var('res', new TensorType(MILDataType.FLOAT32, [1]))];
    b.operations.push(op3);

    const op4 = new Operation(
      'unknown_op',
      { arr: [new Var('arr_in1', new TensorType(MILDataType.FLOAT32, [1]))] },
      [],
    );
    op4.outputs = [];
    b.operations.push(op4);

    const op5 = new Operation('scaled_dot_product_attention', {}, []);
    op5.outputs = [];
    b.operations.push(op5);

    b.outputs = [new Var('block_out')];
    fn.blocks['block0'] = b;
    prog.functions['main'] = fn;

    const converter = new MILToONNXConverter(prog);
    const graph = converter.convert();
    expect(graph.name).toBe('imported_coreml');
  });

  it('handles empty program', () => {
    const prog = new Program();
    const converter = new MILToONNXConverter(prog);
    const graph = converter.convert();
    expect(graph.nodes.length).toBe(0);
  });
});
