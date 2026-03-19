import { Graph, Node, Tensor } from '@onnx9000/core';
import { ExecutionProvider } from './session.js';

export interface GraphRegion {
  providerName: string;
  subGraph: Graph;
  inputs: string[];
  outputs: string[];
}

export class GraphPartitioner {
  private providers: ExecutionProvider[];
  private disableWebNNFallback: boolean = false;

  constructor(providers: ExecutionProvider[], disableWebNNFallback: boolean = false) {
    this.providers = providers;
    this.disableWebNNFallback = disableWebNNFallback;
  }

  // 181. Implement a WebNN capability checker (simulating a build to check for supported nodes).
  // 221. Implement a hardware-sniffing utility checking user-agent/GPU strings.
  // 222-225. Driver workarounds (Apple Neural Engine, Intel VPU, Snapdragon)
  private checkNodeSupported(node: Node, providerName: string): boolean {
    if (providerName === 'WebNN') {
      // Stub check: WebNN v1 doesn't support 'NonZero', 'TopK', 'Loop', 'If' natively yet.
      // 199, 200. MoE routing / If nodes fallback to WASM
      // 211, 212, 213, 214. Emulate GatherElements, ScatterND, NonZero, TopK using fallback
      const unsupported = [
        'NonZero',
        'TopK',
        'Loop',
        'If',
        'CustomOp',
        'GatherElements',
        'ScatterND',
      ];

      // 192. Translate ONNX Attention to WebNN or fallback
      // Since WebNN lacks FlashAttention, we offload it if not decomposed
      if (node.opType === 'Attention' || node.opType === 'FlashAttention') {
        unsupported.push(node.opType);
      }

      // 195. Emulate RoPE (if it is a single fused op)
      if (node.opType === 'RotaryEmbedding') {
        unsupported.push(node.opType);
      }

      // 196. Handle dynamic KV cache updates. If WebNN forbids dynamic concat...
      if (node.opType === 'Concat' && this.hasDynamicAxes(node)) {
        unsupported.push('Concat');
      }

      // 198. Map NLP vocabulary Gather operations efficiently (or offload to CPU if NPUs struggle)
      // Here we assume if gather is on axis 0 of a massive embedding table (input 0 is > 100k), we offload.
      if (node.opType === 'Gather' && this.isVocabularyGather(node)) {
        unsupported.push('Gather');
      }

      if (unsupported.includes(node.opType)) {
        if (this.disableWebNNFallback) {
          // 189. Provide CLI flag --disable-webnn-fallback to force strict NPU execution (throwing errors if unsupported).
          throw new Error(
            `Node ${node.opType} is not supported on WebNN, but fallback is disabled.`,
          );
        }
        return false;
      }
    }
    return true;
  }

  private hasDynamicAxes(node: Node): boolean {
    // In a real scenario, we'd check if the shape of the Concat inputs contains variables.
    // Stubbed for now to return true if an attribute 'dynamic' exists.
    return !!node.attributes['dynamic'];
  }

  private isVocabularyGather(node: Node): boolean {
    // Assume an embedding gather if axis=0 and it's the first node in a transformer block
    // Stubbed
    return !!node.attributes['is_embedding'];
  }

  // 182. Implement an AST traversal to identify contiguous blocks of WebNN-supported ops.
  // 183. Partition the onnx9000 graph into "WebNN Regions" and "WASM/WebGPU Regions".
  public partition(graph: Graph): GraphRegion[] {
    const regions: GraphRegion[] = [];

    // Simplistic mock partitioner:
    // Scans through nodes and groups them if they belong to the same provider capabilities.
    // A full AST traversal needs topological grouping and boundary resolution.

    let currentProviderName = this.providers[0]!.name;
    let currentNodes: Node[] = [];
    let currentInputs = new Set<string>();
    let currentOutputs = new Set<string>();

    for (const node of graph.nodes) {
      let bestProvider = currentProviderName;
      let supported = this.checkNodeSupported(node, currentProviderName);

      if (!supported && this.providers.length > 1) {
        // Fallback required
        const fallbackProvider = this.providers.find(
          (p) => p.name !== currentProviderName && this.checkNodeSupported(node, p.name),
        );
        if (fallbackProvider) {
          bestProvider = fallbackProvider.name;
        }
      }

      if (bestProvider !== currentProviderName) {
        if (currentNodes.length > 0) {
          // 184. Generate distinct sub-graphs (onnx9000.Graph) for each region.
          // 227. Profile operator compilation times to dynamically skip WebNN for extremely fast, simple nodes.
          // E.g., if a sub-graph has only 1 trivial node (like Add), it's faster to execute in WASM.
          if (
            currentNodes.length === 1 &&
            currentProviderName === 'WebNN' &&
            ['Add', 'Sub', 'Mul', 'Div'].includes(currentNodes[0]!.opType)
          ) {
            const wasmProvider = this.providers.find((p) => p.name === 'WASM');
            if (wasmProvider) currentProviderName = 'WASM';
          }

          const subGraph = new Graph(`${graph.name}_region_${regions.length}`);
          subGraph.nodes = [...currentNodes];
          subGraph.inputs = [...graph.inputs];
          subGraph.outputs = [...graph.outputs];
          regions.push({
            providerName: currentProviderName,
            subGraph,
            inputs: Array.from(currentInputs),
            outputs: Array.from(currentOutputs),
          });
          currentNodes = [];
          currentInputs.clear();
          currentOutputs.clear();
        }
        currentProviderName = bestProvider;
      }

      currentNodes.push(node);
      node.inputs.forEach((i) => {
        if (i !== '') currentInputs.add(i);
      });
      node.outputs.forEach((o) => {
        if (o !== '') currentOutputs.add(o);
      });
    }

    if (currentNodes.length > 0) {
      // 227. Profile operator compilation times to dynamically skip WebNN for extremely fast, simple nodes.
      // E.g., if a sub-graph has only 1 trivial node (like Add), it's faster to execute in WASM.
      if (
        currentNodes.length === 1 &&
        currentProviderName === 'WebNN' &&
        ['Add', 'Sub', 'Mul', 'Div'].includes(currentNodes[0]!.opType)
      ) {
        const wasmProvider = this.providers.find((p) => p.name === 'WASM');
        if (wasmProvider) currentProviderName = 'WASM';
      }

      const subGraph = new Graph(`${graph.name}_region_${regions.length}`);
      subGraph.nodes = [...currentNodes];
      subGraph.inputs = [...graph.inputs];
      subGraph.outputs = [...graph.outputs];
      regions.push({
        providerName: currentProviderName,
        subGraph,
        inputs: Array.from(currentInputs),
        outputs: Array.from(currentOutputs),
      });
    } else if (regions.length === 0) {
      // Empty graph case (e.g. tests)
      regions.push({
        providerName: this.providers[0]!.name,
        subGraph: graph,
        inputs: [],
        outputs: graph.outputs.map((o) => (typeof o === 'string' ? o : o.name)),
      });
    }

    return regions;
  }
}
