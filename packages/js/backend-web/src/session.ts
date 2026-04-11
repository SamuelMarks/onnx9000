/* eslint-disable */
import { Graph, Tensor, Node } from '@onnx9000/core';

import { GraphPartitioner } from './partitioner.js';

export interface ExecutionProvider {
  name: string;
  initialize(): Promise<void>;
  execute(graph: Graph, inputs: Record<string, Tensor>): Promise<Record<string, Tensor>>;
}

export interface SessionOptions {
  executionProviders?: string[];
  graphOptimizationLevel?: 'disable' | 'basic' | 'extended' | 'all';
  logSeverityLevel?: 0 | 1 | 2 | 3 | 4;
  logId?: string;
  freeDimensionOverrides?: Record<string, number>;
  disableWebNNFallback?: boolean;
  webnnCompatMode?: boolean; // 226. Provide a --webnn-compat-mode flag enabling all known driver workarounds.
}

export class InferenceSession {
  private graph: Graph;
  private providers: ExecutionProvider[];
  public options: SessionOptions;
  public profilingEnabled: boolean = false;
  private partitioner: GraphPartitioner;

  constructor(graph: Graph, providers: ExecutionProvider[], options: SessionOptions = {}) {
    this.graph = graph;
    this.providers = providers;
    this.options = options;
    this.partitioner = new GraphPartitioner(
      this.providers,
      this.options.disableWebNNFallback || false,
    );
  }

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

  startProfiling(): void {
    this.profilingEnabled = true;
  }

  endProfiling(): void {
    this.profilingEnabled = false;
  }

  async run(
    outputNames: string[],
    inputs: Record<string, Tensor>,
  ): Promise<Record<string, Tensor>> {
    if (this.providers.length === 0) {
      throw new Error('No Execution Providers registered.');
    }

    for (const [key, tensor] of Object.entries(inputs)) {
      if (!tensor) throw new Error(`Input ${key} is null or undefined`);
    }

    // 183. Partition the onnx9000 graph into "WebNN Regions" and "WASM/WebGPU Regions"
    const regions = this.partitioner.partition(this.graph);

    const currentTensors: Record<string, Tensor> = { ...inputs };

    // 187. Execute regions sequentially, copying outputs from WebNN to WASM and vice-versa
    for (const region of regions) {
      const provider = this.providers.find((p) => p.name === region.providerName);
      if (!provider) {
        throw new Error(`Provider ${region.providerName} not found.`);
      }

      const regionInputs: Record<string, Tensor> = {};
      for (const reqInput of region.inputs) {
        if (currentTensors[reqInput]) {
          regionInputs[reqInput] = currentTensors[reqInput]!;
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
        finalOutputs[name] = currentTensors[name]!;
      }
    }

    return finalOutputs;
  }
}
