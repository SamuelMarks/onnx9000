/* eslint-disable */
/**
 * Dynamic batching utilities to optimize singleton execution graphs into batch-enabled graphs natively.
 * @module
 */
import { Block, Var } from './ast.js';
import { TensorType } from './types.js';

/**
 * Sweeps the AST and replaces `1` dimensions at index 0 with a dynamic `B` token.
 * This dynamically unlocks KServe V2 batching protocols for Core ML `.mlpackage` targets natively.
 * @param block - The MIL Block to transform.
 */
export function implementDynamicBatching(block: Block): void {
  // 291. Implement dynamic graph batching
  // If the first dimension of the tensor shape is static 1, we replace it with dynamic -1 (symbolic 'B')

  for (const input of block.inputs) {
    if (input.type instanceof TensorType && input.type.shape[0] === 1) {
      input.type.shape[0] = 'B';
    }
  }

  for (const op of block.operations) {
    for (const out of op.outputs) {
      if (out.type instanceof TensorType && out.type.shape[0] === 1) {
        out.type.shape[0] = 'B';
      }
    }
  }
}
