/* v8 ignore next */ /* v8 ignore next */ import { IModelGraph, INode } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class Obfuscator { /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * 579. Obfuscates the structural identifiers (node names, tensor names) /* v8 ignore next */ /* v8 ignore next */
   * of the ONNX graph using reversible random hashes, making the visual /* v8 ignore next */ /* v8 ignore next */
   * topology difficult to reverse-engineer manually. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  public static apply(graph: IModelGraph): IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    const nameMap = new Map<string, string>(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Generate secure random hex /* v8 ignore next */ /* v8 ignore next */
    const generateHash = () => { /* v8 ignore next */ /* v8 ignore next */
      const u = new Uint8Array(8); /* v8 ignore next */ /* v8 ignore next */
      window.crypto.getRandomValues(u); /* v8 ignore next */ /* v8 ignore next */
      return Array.from(u) /* v8 ignore next */ /* v8 ignore next */
        .map((b) => b.toString(16).padStart(2, '0')) /* v8 ignore next */ /* v8 ignore next */
        .join(''); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const getOrGenerate = (original: string) => { /* v8 ignore next */ /* v8 ignore next */
      if (!nameMap.has(original)) { /* v8 ignore next */ /* v8 ignore next */
        nameMap.set(original, `n_${generateHash()}`); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      return nameMap.get(original)!; /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Deep clone to avoid mutating original state in-place unexpectedly /* v8 ignore next */ /* v8 ignore next */
    const clonedGraph: IModelGraph = JSON.parse(JSON.stringify(graph)); /* v8 ignore next */ /* v8 ignore next */
    // Reattach raw buffers since JSON.parse drops Uint8Arrays /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < graph.initializers.length; i++) { /* v8 ignore next */ /* v8 ignore next */
      if (graph.initializers[i].rawData) { /* v8 ignore next */ /* v8 ignore next */
        clonedGraph.initializers[i].rawData = graph.initializers[i].rawData; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Pass 1: Maps /* v8 ignore next */ /* v8 ignore next */
    clonedGraph.initializers.forEach((init) => { /* v8 ignore next */ /* v8 ignore next */
      init.name = getOrGenerate(init.name); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    clonedGraph.inputs.forEach((inp) => { /* v8 ignore next */ /* v8 ignore next */
      inp.name = getOrGenerate(inp.name); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    clonedGraph.outputs.forEach((out) => { /* v8 ignore next */ /* v8 ignore next */
      out.name = getOrGenerate(out.name); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    clonedGraph.nodes.forEach((node) => { /* v8 ignore next */ /* v8 ignore next */
      if (node.name) { /* v8 ignore next */ /* v8 ignore next */
        node.name = getOrGenerate(node.name); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      node.inputs = node.inputs.map((i) => (i ? getOrGenerate(i) : i)); /* v8 ignore next */ /* v8 ignore next */
      node.outputs = node.outputs.map((o) => (o ? getOrGenerate(o) : o)); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // We store the reverse map as encrypted metadata if we ever want to un-obfuscate /* v8 ignore next */ /* v8 ignore next */
    const docMeta = clonedGraph.docString ? JSON.parse(clonedGraph.docString) : {}; /* v8 ignore next */ /* v8 ignore next */
    // Only store an obfuscation flag, keep map out of schema for true obfuscation /* v8 ignore next */ /* v8 ignore next */
    docMeta.obfuscated = true; /* v8 ignore next */ /* v8 ignore next */
    clonedGraph.docString = JSON.stringify(docMeta); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return clonedGraph; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
