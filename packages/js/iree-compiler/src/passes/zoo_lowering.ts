import { Graph, Node } from '@onnx9000/core';
import { Operation, Region, Block, Type } from '../ir/core';

export class ZooMLIRLoweringPass {
  /**
   * Translates Core IR strictly to linalg and tensor MLIR dialects.
   */
  static lowerToMLIR(graph: Graph): Operation {
    const modRegion = new Region();
    const modBlock = new Block(modRegion);
    modRegion.pushBlock(modBlock);
    const modOp = new Operation('builtin.module', [], [], { sym_name: graph.name }, [modRegion]);

    const funcRegion = new Region();
    const funcBlock = new Block(funcRegion);
    funcRegion.pushBlock(funcBlock);

    // Mock translation mapping
    for (const node of graph.nodes) {
      if (node.opType === 'MatMul') {
        const op = new Operation('linalg.matmul', [], [{ id: 'tensor<?x?xf32>' }], {
          ins: [node.inputs[0], node.inputs[1]],
          outs: ['init'],
        });
        funcBlock.pushOperation(op);
      } else if (node.opType === 'Add') {
        const op = new Operation('linalg.add', [], [{ id: 'tensor<?xf32>' }], {
          ins: [node.inputs[0], node.inputs[1]],
          outs: ['init'],
        });
        funcBlock.pushOperation(op);
      } else {
        const op = new Operation(`unmapped.${node.opType}`, [], [], {});
        funcBlock.pushOperation(op);
      }
    }

    const funcOp = new Operation('func.func', [], [], { sym_name: 'main' }, [funcRegion]);
    modBlock.pushOperation(funcOp);

    return modOp;
  }
}
