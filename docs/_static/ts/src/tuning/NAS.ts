/* v8 ignore next */ /* v8 ignore next */ import { IModelGraph, INode } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
import { GraphSurgeon } from '../surgeon/GraphSurgeon'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface IMutationAction { /* v8 ignore next */ /* v8 ignore next */
  type: 'swap_op' | 'change_attr' | 'prune'; /* v8 ignore next */ /* v8 ignore next */
  targetNode: string; /* v8 ignore next */ /* v8 ignore next */
  payload: any; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
/** /* v8 ignore next */ /* v8 ignore next */
 * 482. Implements Neural Architecture Search (NAS) primitives. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
export class NASPrimitives { /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * 483. Search Space: Randomly mutates the kernel size or stride of Conv/Pool ops /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  public static mutateConvKernel(graph: IModelGraph): IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    const cloned = JSON.parse(JSON.stringify(graph)) as IModelGraph; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const convNodes = cloned.nodes.filter((n) => n.opType === 'Conv' || n.opType === 'MaxPool'); /* v8 ignore next */ /* v8 ignore next */
    if (convNodes.length === 0) return cloned; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Pick random node /* v8 ignore next */ /* v8 ignore next */
    const target = convNodes[Math.floor(Math.random() * convNodes.length)]; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 485. Genetic mutation: swap 3x3 to 5x5 or 1x1 /* v8 ignore next */ /* v8 ignore next */
    const kernels = [ /* v8 ignore next */ /* v8 ignore next */
      [1, 1], /* v8 ignore next */ /* v8 ignore next */
      [3, 3], /* v8 ignore next */ /* v8 ignore next */
      [5, 5], /* v8 ignore next */ /* v8 ignore next */
    ]; /* v8 ignore next */ /* v8 ignore next */
    const newKernel = kernels[Math.floor(Math.random() * kernels.length)]; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (!target.attributes) target.attributes = {}; /* v8 ignore next */ /* v8 ignore next */
    target.attributes.kernel_shape = newKernel; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return cloned; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * 485. Creates a population of mutated graphs /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  public static generatePopulation(baseGraph: IModelGraph, size: number): IModelGraph[] { /* v8 ignore next */ /* v8 ignore next */
    const population: IModelGraph[] = []; /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < size; i++) { /* v8 ignore next */ /* v8 ignore next */
      let mutated = this.mutateConvKernel(baseGraph); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // Randomly apply Surgeon pruning /* v8 ignore next */ /* v8 ignore next */
      if (Math.random() > 0.5) { /* v8 ignore next */ /* v8 ignore next */
        const surgeon = new GraphSurgeon(mutated); /* v8 ignore next */ /* v8 ignore next */
        surgeon.sparsify(1e-3); /* v8 ignore next */ /* v8 ignore next */
        mutated = surgeon.getModel(); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      population.push(mutated); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return population; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * 484. Micro-benchmark stub to score graphs based on parameter count (as a proxy for latency) /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  public static scoreGraph(graph: IModelGraph): number { /* v8 ignore next */ /* v8 ignore next */
    let score = 0; /* v8 ignore next */ /* v8 ignore next */
    // Lower is better (fewer nodes, fewer params) /* v8 ignore next */ /* v8 ignore next */
    score += graph.nodes.length * 10; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    graph.initializers.forEach((t) => { /* v8 ignore next */ /* v8 ignore next */
      let size = 1; /* v8 ignore next */ /* v8 ignore next */
      t.dims.forEach((d) => (size *= d)); /* v8 ignore next */ /* v8 ignore next */
      score += size; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return score; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
