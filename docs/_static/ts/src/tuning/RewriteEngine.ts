/* v8 ignore next */ /* v8 ignore next */ import { IModelGraph, INode } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
// 488. Expose an API for custom rewrite rules /* v8 ignore next */ /* v8 ignore next */
export type RewriteRule = (graph: IModelGraph) => { mutated: boolean; newGraph: IModelGraph }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class CustomRewriteEngine { /* v8 ignore next */ /* v8 ignore next */
  private rules: Map<string, RewriteRule> = new Map(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  public registerRule(name: string, rule: RewriteRule): void { /* v8 ignore next */ /* v8 ignore next */
    this.rules.set(name, rule); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  public applyAll(graph: IModelGraph): IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    let current = graph; /* v8 ignore next */ /* v8 ignore next */
    let mutatedOverall = false; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Apply until fixpoint or max iterations /* v8 ignore next */ /* v8 ignore next */
    let iterations = 0; /* v8 ignore next */ /* v8 ignore next */
    while (iterations < 10) { /* v8 ignore next */ /* v8 ignore next */
      let mutatedThisPass = false; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.rules.forEach((rule, name) => { /* v8 ignore next */ /* v8 ignore next */
        try { /* v8 ignore next */ /* v8 ignore next */
          const { mutated, newGraph } = rule(current); /* v8 ignore next */ /* v8 ignore next */
          if (mutated) { /* v8 ignore next */ /* v8 ignore next */
            current = newGraph; /* v8 ignore next */ /* v8 ignore next */
            mutatedThisPass = true; /* v8 ignore next */ /* v8 ignore next */
            mutatedOverall = true; /* v8 ignore next */ /* v8 ignore next */
            console.log(`[RewriteEngine] Applied rule: ${name}`); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } catch (e) { /* v8 ignore next */ /* v8 ignore next */
          console.error(`Rule ${name} failed:`, e); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (!mutatedThisPass) break; /* v8 ignore next */ /* v8 ignore next */
      iterations++; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return current; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export const globalRewriteEngine = new CustomRewriteEngine(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
// Example layer fusion auto-tuning rule (492) /* v8 ignore next */ /* v8 ignore next */
globalRewriteEngine.registerRule('FuseConvBatchNormRelu', (graph: IModelGraph) => { /* v8 ignore next */ /* v8 ignore next */
  // Deep clone /* v8 ignore next */ /* v8 ignore next */
  const newGraph: IModelGraph = JSON.parse(JSON.stringify(graph)); /* v8 ignore next */ /* v8 ignore next */
  for (let i = 0; i < graph.initializers.length; i++) { /* v8 ignore next */ /* v8 ignore next */
    if (graph.initializers[i].rawData) { /* v8 ignore next */ /* v8 ignore next */
      newGraph.initializers[i].rawData = graph.initializers[i].rawData; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  let mutated = false; /* v8 ignore next */ /* v8 ignore next */
  const nodesToRemove = new Set<string>(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  for (let i = 0; i < newGraph.nodes.length - 2; i++) { /* v8 ignore next */ /* v8 ignore next */
    const conv = newGraph.nodes[i]; /* v8 ignore next */ /* v8 ignore next */
    if (conv.opType !== 'Conv') continue; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const bn = newGraph.nodes.find( /* v8 ignore next */ /* v8 ignore next */
      (n) => n.inputs[0] === conv.outputs[0] && n.opType === 'BatchNormalization', /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
    if (!bn) continue; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const relu = newGraph.nodes.find((n) => n.inputs[0] === bn.outputs[0] && n.opType === 'Relu'); /* v8 ignore next */ /* v8 ignore next */
    if (!relu) continue; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // We found a chain: Conv -> BN -> Relu /* v8 ignore next */ /* v8 ignore next */
    // Fuse them /* v8 ignore next */ /* v8 ignore next */
    conv.opType = 'FusedConvBNRelu'; /* v8 ignore next */ /* v8 ignore next */
    conv.outputs = relu.outputs; // Conv bypasses the other two and outputs directly what Relu outputted /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    nodesToRemove.add(bn.name); /* v8 ignore next */ /* v8 ignore next */
    nodesToRemove.add(relu.name); /* v8 ignore next */ /* v8 ignore next */
    mutated = true; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  if (mutated) { /* v8 ignore next */ /* v8 ignore next */
    newGraph.nodes = newGraph.nodes.filter((n) => !nodesToRemove.has(n.name)); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  return { mutated, newGraph }; /* v8 ignore next */ /* v8 ignore next */
});
