import { Graph, Tensor, Node } from '@onnx9000/core';

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
}

export class InferenceSession {
  private graph: Graph;
  private providers: ExecutionProvider[];
  public options: SessionOptions;
  public profilingEnabled: boolean = false;

  constructor(graph: Graph, providers: ExecutionProvider[], options: SessionOptions = {}) {
    this.graph = graph;
    this.providers = providers;
    this.options = options;
  }

  static async create(
    modelData: string | ArrayBuffer,
    options: SessionOptions = {},
  ): Promise<InferenceSession> {
    // In a real engine, we'd parse the ArrayBuffer into a Graph using protobuf.
    // For this Web Architecture representation, we mock the parsing.
    const g = new Graph('Model');
    if (typeof modelData === 'string') {
      g.name = modelData;
    }

    // Default providers
    let providers: ExecutionProvider[] = [];
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

    // Validate inputs
    for (const [key, tensor] of Object.entries(inputs)) {
      if (!tensor) throw new Error(`Input ${key} is null or undefined`);
    }

    const provider = this.providers[0];
    return await provider!.execute(this.graph, inputs);
  }
}
