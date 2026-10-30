/**
 * @fileoverview index.ts
 * Provides index functionality for the backend-web package.
 */
import { type Graph, Tensor } from '@onnx9000/core';
import type { ExecutionProvider } from '../../session.js';

/** Represents the `WasmProvider` class. */
export class WasmProvider implements ExecutionProvider {
  /** Property `name` of `WasmProvider`. */
  name = 'Wasm';

  /** Executes `initialize` on `WasmProvider`. */
  async initialize(): Promise<void> {
    // WASM initialization
  }

  /**
   * Executes `execute` on `WasmProvider`.
   * @param graph Parameter `graph`.
   * @param _inputs Parameter `_inputs`.
   * @returns Result of `execute`.
   */
  async execute(graph: Graph, _inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    // Wasm execution
    const results: Record<string, Tensor> = {};
    for (const name of graph.outputs) {
      const outName =
        typeof name === 'string' ? name : (name as ReturnType<typeof JSON.parse>).name;
      results[outName] = new Tensor(outName, [1], 'float32');
    }
    return results;
  }
}
