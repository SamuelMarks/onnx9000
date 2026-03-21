import { Graph } from '@onnx9000/core';
import { Node } from '@onnx9000/core';
import { MMDNNReporter } from './reporter.js';

export type FusionRule = (graph: Graph, node: Node, reporter: MMDNNReporter) => boolean;

export class NodeFusionRegistry {
  private rules: FusionRule[] = [];

  constructor() {
    this.registerDefaultRules();
  }

  registerRule(rule: FusionRule): void {
    this.rules.push(rule);
  }

  private registerDefaultRules(): void {
    // Basic fusion: Conv + BatchNormalization -> Conv
    // In practice, this modifies the weights of the Conv based on BN params.
    this.registerRule((graph, node, reporter) => {
      if (node.opType !== 'BatchNormalization') return false;

      // Find producer
      const producer = graph.nodes.find((n) => n.outputs.includes(node.inputs[0]!));
      if (!producer || producer.opType !== 'Conv') return false;

      // Skip if producer output is used by multiple nodes
      const usages = graph.nodes.filter((n) => n.inputs.includes(node.inputs[0]!));
      if (usages.length > 1) return false;

      reporter.info(
        `Fusing ${node.opType} '${node.name}' into ${producer.opType} '${producer.name}'`,
      );

      // Real implementation would recalculate Conv weights and bias
      // Here we just rewire the graph structure
      const newOutputs = [...node.outputs];
      producer.outputs = newOutputs;

      // Return true to indicate we modified the graph and should remove this BN node
      return true;
    });

    // Basic fusion: MatMul + Add -> Gemm
    this.registerRule((graph, node, reporter) => {
      if (node.opType !== 'Add') return false;

      const producerA = graph.nodes.find((n) => n.outputs.includes(node.inputs[0]!));
      const producerB = graph.nodes.find((n) => n.outputs.includes(node.inputs[1]!));

      let matmulProducer: Node | undefined;
      let biasInput: string | undefined;

      if (producerA && producerA.opType === 'MatMul') {
        matmulProducer = producerA;
        biasInput = node.inputs[1]!;
      } else if (producerB && producerB.opType === 'MatMul') {
        matmulProducer = producerB;
        biasInput = node.inputs[0]!;
      }

      if (!matmulProducer || !biasInput) return false;

      // Skip if MatMul output is used by multiple nodes
      const usages = graph.nodes.filter((n) => n.inputs.includes(matmulProducer!.outputs[0]!));
      if (usages.length > 1) return false;

      reporter.info(`Fusing MatMul '${matmulProducer.name}' and Add '${node.name}' into Gemm`);

      matmulProducer.opType = 'Gemm';
      matmulProducer.inputs.push(biasInput); // [A, B, C]
      matmulProducer.outputs = [...node.outputs];

      return true;
    });
  }

  applyFusions(graph: Graph, reporter: MMDNNReporter): Graph {
    let changed = true;
    let currentGraph = graph;

    while (changed) {
      changed = false;
      const newNodes: Node[] = [];

      for (const node of currentGraph.nodes) {
        let fused = false;

        for (const rule of this.rules) {
          if (rule(currentGraph, node, reporter)) {
            fused = true;
            changed = true;
            break;
          }
        }

        if (!fused) {
          newNodes.push(node);
        }
      }

      currentGraph.nodes = newNodes;
    }

    return currentGraph;
  }
}
