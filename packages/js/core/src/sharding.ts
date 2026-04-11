/* eslint-disable */
import { Graph } from './ir/graph.js';
import { Node } from './ir/node.js';
import { Tensor } from './ir/tensor.js';
import { recordOp } from './macros.js';

export class AutoShardingPass {
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

export class SPMDLoweringPass {
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

export function allReduce(x: Tensor, group: string = 'world'): Tensor {
  return recordOp('AllReduce', [x], { group });
}

export function allGather(x: Tensor, group: string = 'world'): Tensor {
  return recordOp('AllGather', [x], { group });
}

export function reduceScatter(x: Tensor, group: string = 'world'): Tensor {
  return recordOp('ReduceScatter', [x], { group });
}

export function allToAll(x: Tensor, group: string = 'world'): Tensor {
  return recordOp('AllToAll', [x], { group });
}
