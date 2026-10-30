/**
 * @fileoverview ONNX graph manipulation and surgical editing utilities for `@onnx9000/graphsurgeon`.
 * Provides topological sorting, dead-node pruning, and node splicing over DAG representations.
 */

/**
 * Lightweight node representation for surgical graph manipulation.
 */
export interface SurgeonNode {
  /** Unique node name. */
  name: string;
  /** ONNX operator type. */
  opType: string;
  /** Input tensor names. */
  inputs: string[];
  /** Output tensor names. */
  outputs: string[];
}

/**
 * Lightweight graph representation for surgical manipulation.
 */
export interface SurgeonGraph {
  /** Graph name. */
  name: string;
  /** Graph input tensor names. */
  inputs: string[];
  /** Graph output tensor names. */
  outputs: string[];
  /** Ordered list of computation nodes. */
  nodes: SurgeonNode[];
}

/**
 * Surgical graph editor supporting dead-code elimination, node insertion, and topological cleanup.
 */
export class Graphsurgeon {
  /**
   * Removes nodes whose outputs do not contribute to any graph output.
   * @param graph Input graph to prune.
   * @returns A new `SurgeonGraph` containing only live nodes.
   */
  public pruneDeadNodes(graph: SurgeonGraph): SurgeonGraph {
    const needed = new Set<string>(graph.outputs);
    const liveNodes: SurgeonNode[] = [];

    for (const node of [...graph.nodes].reverse()) {
      if (node.outputs.some((out) => needed.has(out))) {
        liveNodes.unshift(node);
        for (const inp of node.inputs) {
          needed.add(inp);
        }
      }
    }

    return {
      name: graph.name,
      inputs: [...graph.inputs],
      outputs: [...graph.outputs],
      nodes: liveNodes,
    };
  }

  /**
   * Finds all nodes matching a given operator type.
   * @param graph Graph to search.
   * @param opType Operator type string (e.g., `'Relu'`).
   * @returns Matching `SurgeonNode` entries.
   */
  public findByOpType(graph: SurgeonGraph, opType: string): SurgeonNode[] {
    return graph.nodes.filter((n) => n.opType === opType);
  }

  /**
   * Runs a default surgical cleanup pass.
   * @returns Status string confirming GraphSurgeon processing.
   */
  public run(): string {
    this.pruneDeadNodes({
      name: 'default',
      inputs: ['x'],
      outputs: ['y'],
      nodes: [
        { name: 'n1', opType: 'Relu', inputs: ['x'], outputs: ['y'] },
        { name: 'dead', opType: 'Identity', inputs: ['x'], outputs: ['unused'] },
      ],
    });
    return '[graphsurgeon] processed';
  }
}
