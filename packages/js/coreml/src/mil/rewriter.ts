import { Block, Operation, Var } from './ast.js';
import { TensorType, MILType } from './types.js';

export function replaceOperation(block: Block, oldOp: Operation, newOps: Operation[]): void {
  const idx = block.operations.indexOf(oldOp);
  if (idx === -1) {
    throw new Error(`Operation ${oldOp.opType} not found in block`);
  }

  // Splice the new operations in place of the old one
  block.operations.splice(idx, 1, ...newOps);
}

export function replaceVarUsage(block: Block, oldVar: Var, newVar: Var): void {
  for (const op of block.operations) {
    for (const key in op.inputs) {
      const inputs = op.inputs[key]!;
      if (Array.isArray(inputs)) {
        for (let i = 0; i < inputs.length; i++) {
          if (inputs[i]!.name === oldVar.name) {
            inputs[i] = newVar;
          }
        }
      } else {
        if (inputs.name === oldVar.name) {
          op.inputs[key] = newVar;
        }
      }
    }
  }

  // Also replace in outputs of the block
  for (let i = 0; i < block.outputs.length; i++) {
    if (block.outputs[i]!.name === oldVar.name) {
      block.outputs[i] = newVar;
    }
  }
}

export function inferShapes(block: Block): void {
  const varShapes = new Map<string, (number | string)[]>();

  // Seed with block inputs
  for (const input of block.inputs) {
    if (input.type instanceof TensorType) {
      varShapes.set(input.name, input.type.shape);
    }
  }

  for (const op of block.operations) {
    // Basic shape inference implementation for common ops
    // Real MIL has exhaustive shape inference rules

    if (
      op.opType === 'add' ||
      op.opType === 'sub' ||
      op.opType === 'mul' ||
      op.opType === 'real_div'
    ) {
      const xInput = op.inputs['x'];
      const yInput = op.inputs['y'];
      if (xInput && yInput && !Array.isArray(xInput) && !Array.isArray(yInput)) {
        // Assume broadcast logic or identical shape
        const shapeX = varShapes.get(xInput.name) || [];
        const shapeY = varShapes.get(yInput.name) || [];
        // very basic max length fallback
        const outShape = shapeX.length > shapeY.length ? shapeX : shapeY;

        for (const out of op.outputs) {
          if (out.type instanceof TensorType) {
            out.type.shape = outShape;
            varShapes.set(out.name, outShape);
          }
        }
      }
    } else if (op.opType === 'const') {
      for (const out of op.outputs) {
        if (out.type instanceof TensorType) {
          varShapes.set(out.name, out.type.shape);
        }
      }
    } else {
      // pass-through for unknown ops
      for (const out of op.outputs) {
        if (out.type instanceof TensorType && out.type.shape.length > 0) {
          varShapes.set(out.name, out.type.shape);
        } else {
          // Find a tensor input and copy shape
          for (const key in op.inputs) {
            const input = op.inputs[key];
            if (!Array.isArray(input) && input) {
              const inShape = varShapes.get(input.name);
              if (inShape && out.type instanceof TensorType) {
                out.type.shape = [...inShape];
                varShapes.set(out.name, out.type.shape);
                break;
              }
            }
          }
        }
      }
    }
  }
}
