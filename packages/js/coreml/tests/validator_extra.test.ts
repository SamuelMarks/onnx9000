import { describe, it, expect } from 'vitest';
import { validateBlock } from '../src/mil/validator.js';
import { Block, Operation, Var } from '../src/mil/ast.js';
import { TensorType, MILDataType } from '../src/mil/types.js';

describe('Validator Extra', () => {
  it('covers missing array inputs', () => {
    const b = new Block('b');
    b.operations.push(
      new Operation(
        'concat',
        { inputs: [new Var('missing', new TensorType(MILDataType.FLOAT32, [1]))] },
        [],
      ),
    );
    expect(() => validateBlock(b)).toThrow('Operation input missing is not available in block b');
  });

  it('covers missing single input', () => {
    const b = new Block('b');
    b.operations.push(
      new Operation('add', { x: new Var('missing', new TensorType(MILDataType.FLOAT32, [1])) }, []),
    );
    expect(() => validateBlock(b)).toThrow('Operation input missing is not available in block b');
  });

  it('covers block inputs', () => {
    const b = new Block('b');
    const v = new Var('in1', new TensorType(MILDataType.FLOAT32, [1]));
    b.inputs.push(v);
    b.operations.push(new Operation('add', { x: v }, []));
    expect(() => validateBlock(b)).not.toThrow();
  });
});
