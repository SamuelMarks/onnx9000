/* eslint-disable */
import { Program, Function, Block, Operation } from './ast.js';
import { topologicalSort } from './sort.js';

export function validateMILProgram(program: Program): boolean {
  for (const fnName in program.functions) {
    const fn = program.functions[fnName]!;
    for (const blockName in fn.blocks) {
      const block = fn.blocks[blockName]!;
      validateBlock(block);
    }
  }
  return true;
}

export function validateBlock(block: Block): void {
  // Check acyclic property (Topological sort inherently checks this)
  try {
    topologicalSort(block.operations);
  } catch (e) {
    throw new Error(`Block ${block.name} is not a valid DAG: ${e}`);
  }

  // Ensure all used variables are produced before they are consumed
  const availableVars = new Set<string>();

  // Add inputs
  for (const input of block.inputs) {
    availableVars.add(input.name);
  }

  for (const op of block.operations) {
    // For const ops, outputs are available
    if (op.opType === 'const') {
      op.outputs.forEach((o) => availableVars.add(o.name));
      continue;
    }

    // Check inputs
    for (const key in op.inputs) {
      const inputs = op.inputs[key]!;
      if (Array.isArray(inputs)) {
        for (const inputVar of inputs) {
          if (!availableVars.has(inputVar.name)) {
            throw new Error(
              `Operation input ${inputVar.name} is not available in block ${block.name}`,
            );
          }
        }
      } else {
        if (!availableVars.has(inputs.name)) {
          throw new Error(`Operation input ${inputs.name} is not available in block ${block.name}`);
        }
      }
    }

    // Add outputs to available vars
    for (const out of op.outputs) {
      availableVars.add(out.name);
    }
  }

  // Check block outputs
  for (const out of block.outputs) {
    if (!availableVars.has(out.name)) {
      throw new Error(`Block output ${out.name} is not produced within block ${block.name}`);
    }
  }
}
