/**
 * @fileoverview graph.ts
 * Provides graph functionality for the webnn-polyfill package.
 */

import { Graph } from '@onnx9000/core';
import type { MLGraph } from './interfaces.js';

/** Represents the `PolyfillMLGraph` class. */
export class PolyfillMLGraph implements MLGraph {
  /** Property `onnxGraph` of `PolyfillMLGraph`. */
  public onnxGraph: Graph;

  // 150. Define MLGraph interface class containing the compiled execution payload.
  // 155. Provide a deterministic compilation ID identifying the graph natively.
  /** Property `compilationId` of `PolyfillMLGraph`. */
  public compilationId: string;

  /**
   * Initializes a new `PolyfillMLGraph` instance.
   * @param graph Parameter `graph`.
   */
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
