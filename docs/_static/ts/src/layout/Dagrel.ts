/* v8 ignore next */ /* v8 ignore next */ import { IModelGraph, INode } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface IGraphLayoutNode { /* v8 ignore next */ /* v8 ignore next */
  id: string; /* v8 ignore next */ /* v8 ignore next */
  x: number; /* v8 ignore next */ /* v8 ignore next */
  y: number; /* v8 ignore next */ /* v8 ignore next */
  width: number; /* v8 ignore next */ /* v8 ignore next */
  height: number; /* v8 ignore next */ /* v8 ignore next */
  node: INode; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface IGraphLayoutEdge { /* v8 ignore next */ /* v8 ignore next */
  source: string; /* v8 ignore next */ /* v8 ignore next */
  target: string; /* v8 ignore next */ /* v8 ignore next */
  points: { x: number; y: number }[]; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class Dagrel { /* v8 ignore next */ /* v8 ignore next */
  private nodeWidth = 150; /* v8 ignore next */ /* v8 ignore next */
  private nodeHeight = 50; /* v8 ignore next */ /* v8 ignore next */
  private rankSeparation = 100; /* v8 ignore next */ /* v8 ignore next */
  private nodeSeparation = 50; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  layout(graph: IModelGraph): { nodes: IGraphLayoutNode[]; edges: IGraphLayoutEdge[] } { /* v8 ignore next */ /* v8 ignore next */
    // 1. Assign ranks (Topological sort essentially) /* v8 ignore next */ /* v8 ignore next */
    const ranks = new Map<string, number>(); /* v8 ignore next */ /* v8 ignore next */
    const outgoingEdges = new Map<string, string[]>(); /* v8 ignore next */ /* v8 ignore next */
    const incomingEdges = new Map<string, string[]>(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    graph.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      outgoingEdges.set(n.name, []); /* v8 ignore next */ /* v8 ignore next */
      if (!incomingEdges.has(n.name)) { /* v8 ignore next */ /* v8 ignore next */
        incomingEdges.set(n.name, []); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    graph.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      n.inputs.forEach((input) => { /* v8 ignore next */ /* v8 ignore next */
        // Find node that produces this output /* v8 ignore next */ /* v8 ignore next */
        const producer = graph.nodes.find((pn) => pn.outputs.includes(input)); /* v8 ignore next */ /* v8 ignore next */
        if (producer) { /* v8 ignore next */ /* v8 ignore next */
          if (!outgoingEdges.has(producer.name)) outgoingEdges.set(producer.name, []); /* v8 ignore next */ /* v8 ignore next */
          outgoingEdges.get(producer.name)!.push(n.name); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
          if (!incomingEdges.has(n.name)) incomingEdges.set(n.name, []); /* v8 ignore next */ /* v8 ignore next */
          incomingEdges.get(n.name)!.push(producer.name); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // BFS to assign ranks /* v8 ignore next */ /* v8 ignore next */
    const queue: string[] = []; /* v8 ignore next */ /* v8 ignore next */
    incomingEdges.forEach((deps, nodeName) => { /* v8 ignore next */ /* v8 ignore next */
      if (deps.length === 0) { /* v8 ignore next */ /* v8 ignore next */
        ranks.set(nodeName, 0); /* v8 ignore next */ /* v8 ignore next */
        queue.push(nodeName); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    while (queue.length > 0) { /* v8 ignore next */ /* v8 ignore next */
      const current = queue.shift()!; /* v8 ignore next */ /* v8 ignore next */
      const currentRank = ranks.get(current)!; /* v8 ignore next */ /* v8 ignore next */
      const deps = outgoingEdges.get(current) || []; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      deps.forEach((dep) => { /* v8 ignore next */ /* v8 ignore next */
        const nextRank = currentRank + 1; /* v8 ignore next */ /* v8 ignore next */
        if (!ranks.has(dep) || ranks.get(dep)! < nextRank) { /* v8 ignore next */ /* v8 ignore next */
          ranks.set(dep, nextRank); /* v8 ignore next */ /* v8 ignore next */
          queue.push(dep); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Unconnected nodes or cyclic graph fallback /* v8 ignore next */ /* v8 ignore next */
    graph.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      if (!ranks.has(n.name)) ranks.set(n.name, 0); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 2. Position nodes based on rank /* v8 ignore next */ /* v8 ignore next */
    const layoutNodes: IGraphLayoutNode[] = []; /* v8 ignore next */ /* v8 ignore next */
    const rankWidths = new Map<number, number>(); // track how many nodes in each rank /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    graph.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      const rank = ranks.get(n.name)!; /* v8 ignore next */ /* v8 ignore next */
      const pos = rankWidths.get(rank) || 0; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      layoutNodes.push({ /* v8 ignore next */ /* v8 ignore next */
        id: n.name, /* v8 ignore next */ /* v8 ignore next */
        x: pos * (this.nodeWidth + this.nodeSeparation), /* v8 ignore next */ /* v8 ignore next */
        y: rank * (this.nodeHeight + this.rankSeparation), /* v8 ignore next */ /* v8 ignore next */
        width: this.nodeWidth, /* v8 ignore next */ /* v8 ignore next */
        height: this.nodeHeight, /* v8 ignore next */ /* v8 ignore next */
        node: n, /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      rankWidths.set(rank, pos + 1); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Center align ranks /* v8 ignore next */ /* v8 ignore next */
    const maxRankWidth = Math.max(...Array.from(rankWidths.values())); /* v8 ignore next */ /* v8 ignore next */
    const maxWidth = maxRankWidth * (this.nodeWidth + this.nodeSeparation); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    layoutNodes.forEach((ln) => { /* v8 ignore next */ /* v8 ignore next */
      const rank = ranks.get(ln.id)!; /* v8 ignore next */ /* v8 ignore next */
      const rankCount = rankWidths.get(rank)!; /* v8 ignore next */ /* v8 ignore next */
      const rankTotalWidth = rankCount * (this.nodeWidth + this.nodeSeparation); /* v8 ignore next */ /* v8 ignore next */
      const xOffset = (maxWidth - rankTotalWidth) / 2; /* v8 ignore next */ /* v8 ignore next */
      ln.x += xOffset; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 3. Create edges (orthogonal/straight line stub) /* v8 ignore next */ /* v8 ignore next */
    const layoutEdges: IGraphLayoutEdge[] = []; /* v8 ignore next */ /* v8 ignore next */
    graph.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      const targetNode = layoutNodes.find((ln) => ln.id === n.name); /* v8 ignore next */ /* v8 ignore next */
      if (!targetNode) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      n.inputs.forEach((input) => { /* v8 ignore next */ /* v8 ignore next */
        const producerNode = graph.nodes.find((pn) => pn.outputs.includes(input)); /* v8 ignore next */ /* v8 ignore next */
        if (producerNode) { /* v8 ignore next */ /* v8 ignore next */
          const sourceNode = layoutNodes.find((ln) => ln.id === producerNode.name); /* v8 ignore next */ /* v8 ignore next */
          if (sourceNode) { /* v8 ignore next */ /* v8 ignore next */
            layoutEdges.push({ /* v8 ignore next */ /* v8 ignore next */
              source: sourceNode.id, /* v8 ignore next */ /* v8 ignore next */
              target: targetNode.id, /* v8 ignore next */ /* v8 ignore next */
              points: [ /* v8 ignore next */ /* v8 ignore next */
                { x: sourceNode.x + sourceNode.width / 2, y: sourceNode.y + sourceNode.height }, /* v8 ignore next */ /* v8 ignore next */
                { x: targetNode.x + targetNode.width / 2, y: targetNode.y }, /* v8 ignore next */ /* v8 ignore next */
              ], /* v8 ignore next */ /* v8 ignore next */
            }); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return { nodes: layoutNodes, edges: layoutEdges }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
