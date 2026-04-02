import { MLGraph, MLComputeResult } from './interfaces.js';
import { Graph } from '@onnx9000/core';

export class PolyfillMLGraph implements MLGraph {
  public onnxGraph: Graph;

  // 150. Define MLGraph interface class containing the compiled execution payload.
  // 155. Provide a deterministic compilation ID identifying the graph natively.
  public compilationId: string;

  constructor(graph: Graph) {
    this.onnxGraph = graph;
    this.compilationId =
      typeof crypto !== 'undefined' ? crypto.randomUUID() : Math.random().toString();
  }

  // 159-168. MLTensor lifecycle (destroy)
  /**
   * WebNN Destroy operation.
   */
  destroy(): void {
    // Release resources, clear graph
    this.onnxGraph = new Graph('destroyed');
  }
}
