/* eslint-disable */
import { Graph, Node } from '@onnx9000/core';

export interface ExportConfig {
  task?: string;
  opset?: number;
  device?: 'cpu' | 'wasm' | 'webgpu' | 'webnn';
  cacheDir?: string;
  split?: boolean;
}

export async function exportModel(
  modelId: string,
  outputDir: string,
  config: ExportConfig = {},
): Promise<void> {
  console.log(`Exporting model ${modelId} to ${outputDir} with config:`, config);
}

export interface OptimizeConfig {
  level?: 'O1' | 'O2' | 'O3' | 'O4';
  disableFusion?: boolean;
  optimizeSize?: boolean;
}

function removeUnusedNodes(nodes: Node[], outputs: string[]): Node[] {
  let changed = true;
  let currentNodes = [...nodes];

  while (changed) {
    changed = false;
    const usedInputs = new Set<string>(outputs);
    for (const node of currentNodes) {
      for (const inp of node.inputs) {
        usedInputs.add(inp);
      }
    }

    const newNodes = currentNodes.filter((node) => {
      // Keep node if it produces a used output
      const isUsed = node.outputs.some((out) => usedInputs.has(out));
      if (!isUsed) changed = true;
      return isUsed;
    });

    currentNodes = newNodes;
  }
  return currentNodes;
}

function removeIdentityAndDropout(nodes: Node[], graphOutputs: string[]): Node[] {
  const replacementMap = new Map<string, string>();
  const resolve = (name: string): string => {
    let curr = name;
    while (replacementMap.has(curr)) {
      curr = replacementMap.get(curr)!;
    }
    return curr;
  };

  const newNodes: Node[] = [];
  const graphOutputSet = new Set(graphOutputs);

  for (const node of nodes) {
    if (node.opType === 'Identity' || node.opType === 'Dropout') {
      // For dropout, the first output is the tensor, second is mask (optional)
      const input = resolve(node.inputs[0]!);
      const output = node.outputs[0]!;

      // We can only safely remove identity if its output is not a graph output
      // OR we replace the graph output with the input. But changing graph outputs
      // is tricky. So if it's a graph output, we just keep it unless we rewrite graph outputs.
      // We will assume we can rewrite inputs to other nodes.
      replacementMap.set(output, input);

      if (graphOutputSet.has(output)) {
        // Have to keep it to satisfy output interface
        newNodes.push(new Node(node.opType, [input], node.outputs, node.attributes, node.name));
      }
      continue;
    }

    const newInputs = node.inputs.map(resolve);
    newNodes.push(new Node(node.opType, newInputs, node.outputs, node.attributes, node.name));
  }
  return newNodes;
}

/**
 * Optimizes the ONNX graph by applying structural optimizations like
 * dead code elimination, and fusing redundant nodes.
 * @param graph The source ONNX graph
 * @param config Optimization configuration
 * @returns An optimized ONNX graph
 */
export async function optimize(graph: Graph, config: OptimizeConfig = {}): Promise<Graph> {
  const newGraph = new Graph(graph.name + '_optimized');
  newGraph.inputs = [...graph.inputs];
  newGraph.outputs = [...graph.outputs];
  newGraph.initializers = [...graph.initializers];
  newGraph.tensors = { ...graph.tensors };
  newGraph.valueInfo = [...graph.valueInfo];

  let currentNodes = [...graph.nodes];
  const graphOutputNames = graph.outputs.map((o) => o.name);

  // 1. Remove Identity and Dropout
  currentNodes = removeIdentityAndDropout(currentNodes, graphOutputNames);

  // 2. Dead Code Elimination
  currentNodes = removeUnusedNodes(currentNodes, graphOutputNames);

  // 3. Fusion (Conv + Relu -> ConvRelu pseudo-op for demonstration/optimum-like behavior)
  if (!config.disableFusion) {
    const fusedNodes: Node[] = [];
    const skipSet = new Set<Node>();

    for (let i = 0; i < currentNodes.length; i++) {
      const curr = currentNodes[i]!;
      if (skipSet.has(curr)) continue;

      if (curr.opType === 'Conv') {
        const outName = curr.outputs[0]!;
        // Find next node that uses this output
        const next = currentNodes.find((n) => n.inputs.includes(outName));

        // If it's a Relu and the Conv output is only used by this Relu
        if (
          next &&
          next.opType === 'Relu' &&
          next.inputs[0] === outName &&
          !graphOutputNames.includes(outName)
        ) {
          // Ensure it's the only consumer
          const consumers = currentNodes.filter((n) => n.inputs.includes(outName));
          if (consumers.length === 1) {
            const fused = new Node(
              'ConvRelu',
              curr.inputs,
              next.outputs,
              curr.attributes,
              curr.name + '_fused',
            );
            fusedNodes.push(fused);
            skipSet.add(next);
            continue;
          }
        }
      }

      fusedNodes.push(curr);
    }
    currentNodes = fusedNodes;
  }

  newGraph.nodes = currentNodes;
  return newGraph;
}

/**
 * Simplifies the ONNX graph.
 */
export async function simplify(graph: Graph): Promise<Graph> {
  return optimize(graph, { level: 'O1', disableFusion: true });
}

export interface QuantizeConfig {
  method?: 'dynamic' | 'static';
  gptqBits?: number;
  gptqGroupSize?: number;
}

export async function quantize(graph: Graph, config: QuantizeConfig = {}): Promise<Graph> {
  const newGraph = new Graph(graph.name + '_quantized');
  newGraph.nodes = [...graph.nodes];
  newGraph.inputs = [...graph.inputs];
  newGraph.outputs = [...graph.outputs];
  newGraph.valueInfo = [...graph.valueInfo];
  newGraph.tensors = { ...graph.tensors };
  newGraph.initializers = [...graph.initializers];

  for (const initName of newGraph.initializers) {
    const t = newGraph.tensors[initName];
    if (t && t.dtype === 'float32') {
      t.dtype = 'int8';
    }
  }
  return newGraph;
}

export class Quantizer {
  quantize(model: Graph, config: QuantizeConfig) {
    return quantize(model, config);
  }
}
