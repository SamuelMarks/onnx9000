/**
 * @fileoverview bufferize.ts
 * Provides bufferize functionality for the iree-compiler package.
 */

import * as memref from '../dialects/web/memref.js';
import type { TensorType } from '../dialects/web/tensor.js';
import { Operation, type Region, type Value } from '../ir/core.js';

// 47. Bufferization pass (stub)
/**
 * Executes `bufferizeLinalg`.
 * @param region Parameter `region`.
 */
export function bufferizeLinalg(region: Region): void {
  for (const block of region.blocks) {
    const newOps: Operation[] = [];
    const valueToMemRef = new Map<Value, Value>();

    for (const op of block.operations) {
      if (op.opcode === 'web.tensor.empty') {
        const resType = op.results[0]?.type as TensorType;
        const mType = new memref.MemRefType(resType.shape, resType.elementType);
        const allocOp = memref.alloc(mType);
        newOps.push(allocOp);
        valueToMemRef.set(op.results[0] as Value, allocOp.results[0] as Value);
      } else if (op.opcode === 'web.linalg.fill') {
        const fillVal = op.operands[0] as Value;
        const outTensor = op.operands[1] as Value;
        const outMemRef = valueToMemRef.get(outTensor) || outTensor;

        // Create a memref.store or linalg.fill on memrefs
        const fillMemRefOp = new Operation('web.linalg.fill', [fillVal, outMemRef], []);
        newOps.push(fillMemRefOp);
        valueToMemRef.set(op.results[0] as Value, outMemRef);
      } else if (op.opcode === 'web.linalg.matmul') {
        const lhs = op.operands[0] as Value;
        const rhs = op.operands[1] as Value;
        const out = op.operands[2] as Value;

        const lhsM = valueToMemRef.get(lhs) || lhs;
        const rhsM = valueToMemRef.get(rhs) || rhs;
        const outM = valueToMemRef.get(out) || out;

        const matmulM = new Operation('web.linalg.matmul', [lhsM, rhsM, outM], []);
        newOps.push(matmulM);
        valueToMemRef.set(op.results[0] as Value, outM);
      } else {
        // Generic replacement
        const newOperands = op.operands.map((o) => valueToMemRef.get(o) || o);

        // If it returns a tensor, it becomes an alloc + op into memref
        if (op.results.length > 0 && op.results[0]?.type.id === 'tensor') {
          const resType = op.results[0]?.type as TensorType;
          const mType = new memref.MemRefType(resType.shape, resType.elementType);
          const allocOp = memref.alloc(mType);
          newOps.push(allocOp);

          const memrefOp = new Operation(
            op.opcode,
            [...newOperands, allocOp.results[0] as Value],
            [],
          );
          newOps.push(memrefOp);
          valueToMemRef.set(op.results[0] as Value, allocOp.results[0] as Value);
        } else {
          const mappedOp = new Operation(
            op.opcode,
            newOperands,
            op.results.map((r) => r.type),
            op.attributes,
            op.regions,
          );
          newOps.push(mappedOp);
        }
      }
    }

    block.operations.length = 0;
    for (const newOp of newOps) {
      block.pushOperation(newOp);
    }
  }
}
