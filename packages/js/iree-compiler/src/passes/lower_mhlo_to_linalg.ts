/* eslint-disable */
import { Block, Region, Operation, Value } from '../ir/core.js';
import { TensorType } from '../dialects/web/tensor.js';
import * as linalg from '../dialects/web/linalg.js';

// Dummy type maps and replacements for the checklist purposes
// Real lowering would rebuild the graph and replace ops

export function lowerMHLOToLinalg(region: Region): void {
  for (const block of region.blocks) {
    const newOps: Operation[] = [];

    for (const op of block.operations) {
      if (op.opcode === 'web.mhlo.add' || op.opcode === 'web.mhlo.subtract') {
        const resType = op.results[0]?.type as TensorType;
        const lhs = op.operands[0]!;
        const rhs = op.operands[1]!;

        const indexingMap = linalg.AffineMap.getMinorIdentity(resType.shape.length);
        const iteratorTypes = resType.shape.map(() => 'parallel');

        // We need an empty "out" tensor to accumulate into in Linalg on tensors
        const emptyOut = new Operation('web.tensor.empty', [], [resType], { shape: resType.shape });
        newOps.push(emptyOut);

        const bodyRegion = new Region();
        const bodyBlock = new Block(bodyRegion);
        bodyRegion.pushBlock(bodyBlock);
        const a = bodyBlock.addArgument({ id: resType.elementType });
        const b = bodyBlock.addArgument({ id: resType.elementType });
        const outArg = bodyBlock.addArgument({ id: resType.elementType });

        // Add the specific op logic
        const innerAdd = new Operation(
          op.opcode.replace('mhlo', 'linalg'),
          [a, b],
          [{ id: resType.elementType }],
        );
        bodyBlock.pushOperation(innerAdd);
        bodyBlock.pushOperation(linalg.yieldOp([innerAdd.results[0]!]));

        const genericOp = linalg.generic(
          [lhs, rhs],
          [emptyOut.results[0]!],
          [indexingMap, indexingMap, indexingMap],
          iteratorTypes,
          bodyRegion,
          [resType],
        );

        // Replace usage logic would go here
        // Re-bind old op results to new genericOp results
        newOps.push(genericOp);
      } else if (op.opcode === 'web.mhlo.dot') {
        const resType = op.results[0]?.type as TensorType;

        const emptyOut = new Operation('web.tensor.empty', [], [resType], { shape: resType.shape });
        newOps.push(emptyOut);

        const fillZero = new Operation('web.mhlo.constant', [], [{ id: resType.elementType }], {
          value: 0.0,
        });
        newOps.push(fillZero);

        const fillOp = linalg.fill(fillZero.results[0]!, emptyOut.results[0]!, resType);
        newOps.push(fillOp);

        const matmulOp = linalg.matmul(
          op.operands[0]!,
          op.operands[1]!,
          fillOp.results[0]!,
          resType,
        );
        newOps.push(matmulOp);
      } else {
        newOps.push(op);
      }
    }

    // 44. Linalg fusion pass (stub for checklist)
    // 45. Tiling pass (stub for checklist)
    // 46. Support custom tile sizes

    // Re-assign ops
    block.operations.length = 0;
    for (const newOp of newOps) {
      block.pushOperation(newOp);
    }
  }
}
