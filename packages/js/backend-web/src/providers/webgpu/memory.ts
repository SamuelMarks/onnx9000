import { Graph, Node } from '@onnx9000/core';

export class WebGPUMemoryManager {
  // Current typical WebGPU limits: max buffer size 256MB on some, 2GB on others
  private maxBufferSize: number = 256 * 1024 * 1024;

  constructor(maxBufferSize?: number) {
    if (maxBufferSize) {
      this.maxBufferSize = maxBufferSize;
    }
  }

  checkAndPartition(graph: Graph): Graph[] {
    let currentSize = 0;
    let subgraphs: Graph[] = [];
    let currentNodes: Node[] = [];

    // Simplistic graph chunking logic for demonstration.
    // It checks node weights size heuristically and breaks it into chunks.

    for (const node of graph.nodes) {
      let nodeSize = this.estimateNodeMemory(node);
      if (currentSize + nodeSize > this.maxBufferSize) {
        const subGraph = new Graph(graph.name + '_part' + subgraphs.length);
        subGraph.nodes = currentNodes;
        subgraphs.push(subGraph);
        currentNodes = [];
        currentSize = 0;
      }
      currentNodes.push(node);
      currentSize += nodeSize;
    }

    if (currentNodes.length > 0) {
      const subGraph = new Graph(graph.name + '_part' + subgraphs.length);
      subGraph.nodes = currentNodes;
      subgraphs.push(subGraph);
    }

    return subgraphs;
  }

  private estimateNodeMemory(node: Node): number {
    // Very rough estimate - in a real engine we'd inspect the tensors attached to node attributes/initializers.
    return 10 * 1024 * 1024; // 10MB per node mock
  }
}
