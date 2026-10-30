/**
 * @fileoverview validator.ts
 * Provides validator functionality for the coreml package.
 */
import { Onnx9000Error } from '@onnx9000/core';
import type { Block, Program } from './ast.js';
import { topologicalSort } from './sort.js';

/**
 * Executes `validateMILProgram`.
 * @param program Parameter `program`.
 * @returns Result of `validateMILProgram`.
 */
export function validateMILProgram(program: Program): boolean {
  for (const fn of Object.values(program.functions)) {
    for (const block of Object.values(fn.blocks)) {
      validateBlock(block);
    }
  }
  return true;
}

/**
 * Executes `validateBlock`.
 * @param block Parameter `block`.
 */
export function validateBlock(block: Block): void {
  // Check acyclic property (Topological sort inherently checks this)
  try {
    topologicalSort(block.operations);
  } catch (e) {
    throw new Onnx9000Error(`Block ${block.name} is not a valid DAG: ${e}`);
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
      op.outputs.forEach((o) => {
        availableVars.add(o.name);
      });
      continue;
    }

    // Check inputs
    for (const inputs of Object.values(op.inputs)) {
      if (Array.isArray(inputs)) {
        for (const inputVar of inputs) {
          if (!availableVars.has(inputVar.name)) {
            throw new Onnx9000Error(
              `Operation input ${inputVar.name} is not available in block ${block.name}`,
            );
          }
        }
      } else {
        if (!availableVars.has(inputs.name)) {
          throw new Onnx9000Error(
            `Operation input ${inputs.name} is not available in block ${block.name}`,
          );
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
      throw new Onnx9000Error(
        `Block output ${out.name} is not produced within block ${block.name}`,
      );
    }
  }
}
