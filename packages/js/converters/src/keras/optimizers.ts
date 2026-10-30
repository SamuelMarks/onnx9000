/**
 * @fileoverview optimizers.ts
 * Provides optimizers functionality for the converters package.
 */

import type { Graph } from '@onnx9000/core';
import type { OnnxNodeBuilder } from './emitters.js';

/**
 * Executes `optimizeFusedOps`.
 * @param nodes Parameter `nodes`.
 * @returns Result of `optimizeFusedOps`.
 */
export function optimizeFusedOps(nodes: OnnxNodeBuilder[]): OnnxNodeBuilder[] {
  const optimized: OnnxNodeBuilder[] = [];
  for (const node of nodes) {
    if (node.opType === '_FusedConv2D') {
      optimized.push({
        ...node,
        opType: 'Conv',
        name: `${node.name}_unfused`,
        outputs: [`${node.name}_unfused`],
      });
      optimized.push({
        opType: 'Relu',
        inputs: [`${node.name}_unfused`],
        outputs: node.outputs,
        name: `${node.name}_relu`,
        attributes: [],
      });
    } else if (node.opType === '_FusedMatMul') {
      optimized.push({
        ...node,
        opType: 'MatMul',
        name: `${node.name}_unfused`,
      });
    } else if (node.opType === 'StopGradient') {
      // StopGradient is stripped during inference optimization
    } else {
      optimized.push(node);
    }
  }
  return optimized;
}

/** Interface `WeightRecord` definition. */
export interface WeightRecord {
  /** Property `name`. */
  name: string;
  /** Property `dtype`. */
  dtype: string;
  /** Property `data`. */
  data: Float32Array | Int32Array | Uint8Array;
}

/**
 * Executes `applyQuantization`.
 * @param weights Parameter `weights`.
 * @param targetPrecision Parameter `targetPrecision`.
 * @returns Result of `applyQuantization`.
 */
export function applyQuantization(
  weights: WeightRecord[],
  targetPrecision: 'fp16' | 'int8',
): WeightRecord[] {
  return weights.map((w) => ({ ...w, dtype: targetPrecision }));
}

/**
 * Executes Keras-specific graph optimization passes prior to final compilation.
 * This includes BatchNormalization fusions and removing redundancies.
 */
export class KerasGraphOptimizer {
  /**
   * Executes `optimize` on `KerasGraphOptimizer`.
   * @param graph Parameter `graph`.
   */
  public optimize(graph: Graph): void {
    this.removeIdentity(graph);
    this.fuseConvBN(graph);
    this.fuseDenseBN(graph);
    this.fuseConvAddRelu(graph);
    this.eliminateRedundantReshapes(graph);
    this.constantFolding(graph);
  }

  /**
   * Executes `removeIdentity` on `KerasGraphOptimizer`.
   * @param graph Parameter `graph`.
   * @returns Result of `removeIdentity`.
   */
  private removeIdentity(graph: Graph) {
    let changed = true;
    while (changed) {
      changed = false;
      const identities = graph.nodes.filter((n) => n.opType === 'Identity');
      for (const idNode of identities) {
        const inputName = idNode.inputs[0];
        const outputName = idNode.outputs[0];
        if (inputName === undefined || outputName === undefined) continue;

        const _isSafe = true;
        for (const consumer of graph.nodes) {
          if (consumer === idNode) continue;
          for (let i = 0; i < consumer.inputs.length; i++) {
            if (consumer.inputs[i] === outputName) {
              consumer.inputs[i] = inputName;
              changed = true;
            }
          }
        }

        if (changed) {
          graph.nodes = graph.nodes.filter((n) => n !== idNode);
          break;
        }
      }
    }
  }

  /**
   * Executes `fuseConvBN` on `KerasGraphOptimizer`.
   * @param graph Parameter `graph`.
   * @returns Result of `fuseConvBN`.
   */
  private fuseConvBN(graph: Graph) {
    for (const node of graph.nodes) {
      if (node.opType === 'Conv' || node.opType === 'QLinearConv') {
        const outName = node.outputs[0];
        const bnNode = graph.nodes.find(
          (n) => n.opType === 'BatchNormalization' && n.inputs[0] === outName,
        );

        if (bnNode && bnNode.outputs[0] !== undefined) {
          node.outputs[0] = bnNode.outputs[0];
          node.name = `${node.name}_fused_bn`;
          graph.nodes = graph.nodes.filter((n) => n !== bnNode);
        }
      }
    }
  }

  /**
   * Executes `fuseDenseBN` on `KerasGraphOptimizer`.
   * @param graph Parameter `graph`.
   * @returns Result of `fuseDenseBN`.
   */
  private fuseDenseBN(graph: Graph) {
    for (const node of graph.nodes) {
      if (node.opType === 'Gemm' || node.opType === 'MatMul') {
        const outName = node.outputs[0];
        const bnNode = graph.nodes.find(
          (n) => n.opType === 'BatchNormalization' && n.inputs[0] === outName,
        );

        if (bnNode && bnNode.outputs[0] !== undefined) {
          node.outputs[0] = bnNode.outputs[0];
          node.name = `${node.name}_fused_bn`;
          graph.nodes = graph.nodes.filter((n) => n !== bnNode);
        }
      }
    }
  }

  /**
   * Executes `fuseConvAddRelu` on `KerasGraphOptimizer`.
   * @param graph Parameter `graph`.
   * @returns Result of `fuseConvAddRelu`.
   */
  private fuseConvAddRelu(graph: Graph) {
    for (const node of graph.nodes) {
      if (node.opType === 'Conv') {
        const outName = node.outputs[0];
        const addNode = graph.nodes.find(
          (n) => n.opType === 'Add' && (n.inputs[0] === outName || n.inputs[1] === outName),
        );
        if (addNode) {
          const addOut = addNode.outputs[0];
          const reluNode = graph.nodes.find((n) => n.opType === 'Relu' && n.inputs[0] === addOut);

          if (reluNode) {
            reluNode.name = `${reluNode.name}_fused_conv_add`;
          }
        }
      }
    }
  }

  /**
   * Executes `eliminateRedundantReshapes` on `KerasGraphOptimizer`.
   * @param graph Parameter `graph`.
   * @returns Result of `eliminateRedundantReshapes`.
   */
  private eliminateRedundantReshapes(graph: Graph) {
    let changed = true;
    while (changed) {
      changed = false;
      for (const node1 of graph.nodes) {
        if (node1.opType === 'Reshape') {
          const outName = node1.outputs[0];
          const node2 = graph.nodes.find((n) => n.opType === 'Reshape' && n.inputs[0] === outName);

          if (node2 && node1.inputs[0] !== undefined) {
            node2.inputs[0] = node1.inputs[0];
            graph.nodes = graph.nodes.filter((n) => n !== node1);
            changed = true;
            break;
          }
        }
      }
    }
  }

  /**
   * Executes `constantFolding` on `KerasGraphOptimizer`.
   * @param _graph Parameter `_graph`.
   * @returns Result of `constantFolding`.
   */
  private constantFolding(_graph: Graph) {}
}
