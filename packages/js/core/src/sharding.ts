/**
 * @fileoverview sharding.ts
 * Provides sharding functionality for the core package.
 */
import type { Graph } from './ir/graph.js';
import type { Node } from './ir/node.js';
import type { Tensor } from './ir/tensor.js';
import { recordOp } from './macros.js';

/** Represents the `AutoShardingPass` class. */
export class AutoShardingPass {
  /**
   * Executes `apply` on `AutoShardingPass`.
   * @param graph Parameter `graph`.
   * @returns Result of `apply`.
   */
  apply(graph: Graph): Graph {
    for (const node of graph.nodes) {
      if (node.opType === 'MatMul') {
        if (node.inputs.length >= 2) {
          const shardingA = (node.inputs[0] as ReturnType<typeof JSON.parse>).sharding;
          const shardingB = (node.inputs[1] as ReturnType<typeof JSON.parse>).sharding;
          if (shardingA || shardingB) {
            if (node.outputs.length > 0) {
              (node.outputs[0] as ReturnType<typeof JSON.parse>).sharding = Array(
                (node.outputs[0] as ReturnType<typeof JSON.parse>).shape?.length || 0,
              ).fill(null);
            }
          }
        }
      }
    }
    return graph;
  }
}

/** Represents the `SPMDLoweringPass` class. */
export class SPMDLoweringPass {
  /**
   * Executes `apply` on `SPMDLoweringPass`.
   * @param graph Parameter `graph`.
   * @returns Result of `apply`.
   */
  apply(graph: Graph): Graph {
    const newNodes: Node[] = [];
    for (const node of graph.nodes) {
      if (node.opType === 'MatMul') {
        newNodes.push(node);
      } else {
        newNodes.push(node);
      }
    }
    return graph;
  }
}

/**
 * Executes `allReduce`.
 * @param x Parameter `x`.
 * @param group Parameter `group`.
 * @returns Result of `allReduce`.
 */
export function allReduce(x: Tensor, group: string = 'world'): Tensor {
  return recordOp('AllReduce', [x], { group });
}

/**
 * Executes `allGather`.
 * @param x Parameter `x`.
 * @param group Parameter `group`.
 * @returns Result of `allGather`.
 */
export function allGather(x: Tensor, group: string = 'world'): Tensor {
  return recordOp('AllGather', [x], { group });
}

/**
 * Executes `reduceScatter`.
 * @param x Parameter `x`.
 * @param group Parameter `group`.
 * @returns Result of `reduceScatter`.
 */
export function reduceScatter(x: Tensor, group: string = 'world'): Tensor {
  return recordOp('ReduceScatter', [x], { group });
}

/**
 * Executes `allToAll`.
 * @param x Parameter `x`.
 * @param group Parameter `group`.
 * @returns Result of `allToAll`.
 */
export function allToAll(x: Tensor, group: string = 'world'): Tensor {
  return recordOp('AllToAll', [x], { group });
}
