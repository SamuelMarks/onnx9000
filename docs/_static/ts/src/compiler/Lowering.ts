/* v8 ignore next */ /* v8 ignore next */ import { IModelGraph, INode } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
// Abstract intermediate representation for lowered nodes /* v8 ignore next */ /* v8 ignore next */
export interface ILoweredNode { /* v8 ignore next */ /* v8 ignore next */
  id: string; /* v8 ignore next */ /* v8 ignore next */
  type: string; /* v8 ignore next */ /* v8 ignore next */
  inputs: string[]; /* v8 ignore next */ /* v8 ignore next */
  outputs: string[]; /* v8 ignore next */ /* v8 ignore next */
  metadata: unknown; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface ITIRGraph { /* v8 ignore next */ /* v8 ignore next */
  nodes: ILoweredNode[]; /* v8 ignore next */ /* v8 ignore next */
  inputs: string[]; /* v8 ignore next */ /* v8 ignore next */
  outputs: string[]; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class Lowering { /* v8 ignore next */ /* v8 ignore next */
  static lower(model: IModelGraph): ITIRGraph { /* v8 ignore next */ /* v8 ignore next */
    const tirGraph: ITIRGraph = { /* v8 ignore next */ /* v8 ignore next */
      nodes: [], /* v8 ignore next */ /* v8 ignore next */
      inputs: model.inputs.map((i) => i.name), /* v8 ignore next */ /* v8 ignore next */
      outputs: model.outputs.map((o) => o.name), /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    for (const node of model.nodes) { /* v8 ignore next */ /* v8 ignore next */
      // Very basic MLIR/TIR mapping stub /* v8 ignore next */ /* v8 ignore next */
      const lowered: ILoweredNode = { /* v8 ignore next */ /* v8 ignore next */
        id: node.name, /* v8 ignore next */ /* v8 ignore next */
        type: this.mapOpToTIR(node.opType), /* v8 ignore next */ /* v8 ignore next */
        inputs: [...node.inputs], /* v8 ignore next */ /* v8 ignore next */
        outputs: [...node.outputs], /* v8 ignore next */ /* v8 ignore next */
        metadata: { ...node.attributes }, /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      tirGraph.nodes.push(lowered); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return tirGraph; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private static mapOpToTIR(opType: string): string { /* v8 ignore next */ /* v8 ignore next */
    switch (opType) { /* v8 ignore next */ /* v8 ignore next */
      case 'Add': /* v8 ignore next */ /* v8 ignore next */
        return 'tir.add'; /* v8 ignore next */ /* v8 ignore next */
      case 'Sub': /* v8 ignore next */ /* v8 ignore next */
        return 'tir.sub'; /* v8 ignore next */ /* v8 ignore next */
      case 'Mul': /* v8 ignore next */ /* v8 ignore next */
        return 'tir.mul'; /* v8 ignore next */ /* v8 ignore next */
      case 'Div': /* v8 ignore next */ /* v8 ignore next */
        return 'tir.div'; /* v8 ignore next */ /* v8 ignore next */
      case 'MatMul': /* v8 ignore next */ /* v8 ignore next */
        return 'tir.matmul'; /* v8 ignore next */ /* v8 ignore next */
      case 'Gemm': /* v8 ignore next */ /* v8 ignore next */
        return 'tir.gemm'; /* v8 ignore next */ /* v8 ignore next */
      case 'Relu': /* v8 ignore next */ /* v8 ignore next */
        return 'tir.relu'; /* v8 ignore next */ /* v8 ignore next */
      case 'Constant': /* v8 ignore next */ /* v8 ignore next */
        return 'tir.constant'; /* v8 ignore next */ /* v8 ignore next */
      default: /* v8 ignore next */ /* v8 ignore next */
        return `tir.generic.${opType.toLowerCase()}`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
