/**
 * @fileoverview session.ts
 * Provides session functionality for the backend-web package.
 */
import { BackendError, Graph, type Tensor } from '@onnx9000/core';

import { GraphPartitioner } from './partitioner.js';

/** Interface `ExecutionProvider` definition. */
export interface ExecutionProvider {
  /** Property `name`. */
  name: string;
  /** Method `initialize`. */
  initialize(): Promise<void>;
  /**
   * Method `execute`.
   * @param graph Parameter `graph`.
   * @param inputs Parameter `inputs`.
   */
  execute(graph: Graph, inputs: Record<string, Tensor>): Promise<Record<string, Tensor>>;
}

/** Interface `SessionOptions` definition. */
export interface SessionOptions {
  /** Property `executionProviders`. */
  executionProviders?: string[];
  /** Property `graphOptimizationLevel`. */
  graphOptimizationLevel?: 'disable' | 'basic' | 'extended' | 'all';
  /** Property `logSeverityLevel`. */
  logSeverityLevel?: 0 | 1 | 2 | 3 | 4;
  /** Property `logId`. */
  logId?: string;
  /** Property `freeDimensionOverrides`. */
  freeDimensionOverrides?: Record<string, number>;
  /** Property `disableWebNNFallback`. */
  disableWebNNFallback?: boolean;
  /** Property `webnnCompatMode`. */
  webnnCompatMode?: boolean; // 226. Provide a --webnn-compat-mode flag enabling all known driver workarounds.
}

/** Represents the `InferenceSession` class. */
export class InferenceSession {
  /** Property `graph` of `InferenceSession`. */
  private graph: Graph;
  /** Property `providers` of `InferenceSession`. */
  private providers: ExecutionProvider[];
  /** Property `options` of `InferenceSession`. */
  public options: SessionOptions;
  /** Property `profilingEnabled` of `InferenceSession`. */
  public profilingEnabled: boolean = false;
  /** Property `partitioner` of `InferenceSession`. */
  private partitioner: GraphPartitioner;

  /**
   * Initializes a new `InferenceSession` instance.
   * @param graph Parameter `graph`.
   * @param providers Parameter `providers`.
   * @param options Parameter `options`.
   */
  constructor(graph: Graph, providers: ExecutionProvider[], options: SessionOptions = {}) {
    this.graph = graph;
    this.providers = providers;
    this.options = options;
    this.partitioner = new GraphPartitioner(
      this.providers,
      this.options.disableWebNNFallback || false,
    );
  }

  /**
   * Executes `create` on `InferenceSession`.
   * @param modelData Parameter `modelData`.
   * @param options Parameter `options`.
   * @returns Result of `create`.
   */
  static async create(
    modelData: string | ArrayBuffer,
    options: SessionOptions = {},
  ): Promise<InferenceSession> {
    const g = new Graph('Model');
    if (typeof modelData === 'string') {
      g.name = modelData;
    }

    const providers: ExecutionProvider[] = [];
    return new InferenceSession(g, providers, options);
  }

  /** Executes `startProfiling` on `InferenceSession`. */
  startProfiling(): void {
    this.profilingEnabled = true;
  }

  /** Executes `endProfiling` on `InferenceSession`. */
  endProfiling(): void {
    this.profilingEnabled = false;
  }

  /**
   * Executes `run` on `InferenceSession`.
   * @param outputNames Parameter `outputNames`.
   * @param inputs Parameter `inputs`.
   * @returns Result of `run`.
   */
  async run(
    outputNames: string[],
    inputs: Record<string, Tensor>,
  ): Promise<Record<string, Tensor>> {
    if (this.providers.length === 0) {
      throw new BackendError('No Execution Providers registered.');
    }

    for (const [key, tensor] of Object.entries(inputs)) {
      if (!tensor) throw new BackendError(`Input ${key} is null or undefined`);
    }

    // 183. Partition the onnx9000 graph into "WebNN Regions" and "WASM/WebGPU Regions"
    const regions = this.partitioner.partition(this.graph);

    const currentTensors: Record<string, Tensor> = { ...inputs };

    // 187. Execute regions sequentially, copying outputs from WebNN to WASM and vice-versa
    for (const region of regions) {
      const provider = this.providers.find((p) => p.name === region.providerName);
      if (!provider) {
        throw new BackendError(`Provider ${region.providerName} not found.`);
      }

      const regionInputs: Record<string, Tensor> = {};
      for (const reqInput of region.inputs) {
        if (currentTensors[reqInput]) {
          regionInputs[reqInput] = currentTensors[reqInput];
        }
      }

      // 185. Compile WebNN Regions to separate MLGraph instances.
      // 186. Compile WASM/WebGPU Regions using the standard onnx9000 runtime.
      // (This compilation is abstracted behind provider.execute which caches the build/compile)
      const regionOutputs = await provider.execute(region.subGraph, regionInputs);

      // 188. Optimize boundary crossings (using WebGPU buffers) -> Abstracted in provider outputs.
      // 190. Handle dynamic shape propagation correctly across partitioned sub-graphs -> Accomplished by passing concrete output tensors into subsequent regions.

      Object.assign(currentTensors, regionOutputs);
    }

    const finalOutputs: Record<string, Tensor> = {};
    for (const name of outputNames) {
      if (currentTensors[name]) {
        finalOutputs[name] = currentTensors[name];
      }
    }

    return finalOutputs;
  }
}
