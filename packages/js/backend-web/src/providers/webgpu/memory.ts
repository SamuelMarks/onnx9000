/**
 * @fileoverview memory.ts
 * Provides memory functionality for the backend-web package.
 */
import { Graph, type Node } from '@onnx9000/core';

/** Represents the `WebGPUMemoryManager` class. */
export class WebGPUMemoryManager {
  // Current typical WebGPU limits: max buffer size 256MB on some, 2GB on others
  /** Property `maxBufferSize` of `WebGPUMemoryManager`. */
  private maxBufferSize: number = 256 * 1024 * 1024;

  /**
   * Initializes a new `WebGPUMemoryManager` instance.
   * @param maxBufferSize Parameter `maxBufferSize`.
   */
  constructor(maxBufferSize?: number) {
    if (maxBufferSize) {
      this.maxBufferSize = maxBufferSize;
    }
  }

  /**
   * Executes `checkAndPartition` on `WebGPUMemoryManager`.
   * @param graph Parameter `graph`.
   * @returns Result of `checkAndPartition`.
   */
  checkAndPartition(graph: Graph): Graph[] {
    let currentSize = 0;
    const subgraphs: Graph[] = [];
    let currentNodes: Node[] = [];

    // Simplistic graph chunking logic for demonstration.
    // It checks node weights size heuristically and breaks it into chunks.

    for (const node of graph.nodes) {
      const nodeSize = this.estimateNodeMemory(node);
      if (currentSize + nodeSize > this.maxBufferSize) {
        const subGraph = new Graph(`${graph.name}_part${subgraphs.length}`);
        subGraph.nodes = currentNodes;
        subgraphs.push(subGraph);
        currentNodes = [];
        currentSize = 0;
      }
      currentNodes.push(node);
      currentSize += nodeSize;
    }

    if (currentNodes.length > 0) {
      const subGraph = new Graph(`${graph.name}_part${subgraphs.length}`);
      subGraph.nodes = currentNodes;
      subgraphs.push(subGraph);
    }

    return subgraphs;
  }

  /**
   * Executes `estimateNodeMemory` on `WebGPUMemoryManager`.
   * @param _node Parameter `_node`.
   * @returns Result of `estimateNodeMemory`.
   */
  private estimateNodeMemory(_node: Node): number {
    // Very rough estimate - in a real engine we'd inspect the tensors attached to node attributes/initializers.
    return 10 * 1024 * 1024; // 10MB per node mock
  }
}
