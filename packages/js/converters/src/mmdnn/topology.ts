import { Graph } from '@onnx9000/core';
import { Node } from '@onnx9000/core';
import { MMDNNReporter } from './reporter.js';

/**
 * Topologically sorts the nodes of an ONNX graph to ensure inputs are ready before being consumed.
 * @param graph The ONNX IR Graph
 * @param reporter Reporter instance
 * @returns A new graph with sorted nodes
 */
export function topologicalSort(graph: Graph, reporter: MMDNNReporter): Graph {
  const sorted: Node[] = [];
  const visited = new Set<string>();
  const visiting = new Set<string>();

  const nodeMap = new Map<string, Node>();
  const outputToNodeMap = new Map<string, Node>();

  for (const node of graph.nodes) {
    nodeMap.set(node.name, node);
    for (const out of node.outputs) {
      outputToNodeMap.set(out, node);
    }
  }

  // Pre-fill initializers and graph inputs as available tensors
  const availableTensors = new Set<string>();
  for (const init of graph.initializers) {
    availableTensors.add(init);
  }
  for (const t of Object.keys(graph.tensors)) {
    availableTensors.add(t);
  }
  for (const inp of graph.inputs) {
    availableTensors.add(inp.name);
  }

  function dfs(node: Node) {
    if (visited.has(node.name)) return;
    if (visiting.has(node.name)) {
      reporter.error(`Cyclic graph detected at node: ${node.name}`, node.name);
    }
    visiting.add(node.name);

    for (const inp of node.inputs) {
      if (!availableTensors.has(inp)) {
        const producer = outputToNodeMap.get(inp);
        if (producer) {
          dfs(producer);
        } else if (inp !== '') {
          // If empty string, it's an optional parameter
          reporter.warn(
            `Input tensor ${inp} not found in graph initializers or outputs. It might be an implicit shape.`,
            node.name,
          );
        }
      }
    }

    visiting.delete(node.name);
    visited.add(node.name);
    sorted.push(node);

    // Once visited, this node's outputs are available
    for (const out of node.outputs) {
      availableTensors.add(out);
    }
  }

  for (const node of graph.nodes) {
    if (!visited.has(node.name)) {
      dfs(node);
    }
  }

  const sortedGraph = new Graph(graph.name);
  sortedGraph.id = graph.id;
  sortedGraph.tensors = graph.tensors;
  sortedGraph.inputs = graph.inputs;
  sortedGraph.outputs = graph.outputs;
  sortedGraph.valueInfo = graph.valueInfo;
  sortedGraph.initializers = graph.initializers;
  sortedGraph.opsetImports = graph.opsetImports;
  sortedGraph.nodes = sorted;

  return sortedGraph;
}
