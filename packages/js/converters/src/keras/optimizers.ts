/* eslint-disable */
// @ts-nocheck
import { OnnxNodeBuilder } from './emitters.js';

export function optimizeFusedOps(nodes: OnnxNodeBuilder[]): OnnxNodeBuilder[] {
  const optimized: OnnxNodeBuilder[] = [];
  for (const node of nodes) {
    if (node.opType === '_FusedConv2D') {
      optimized.push({
        ...node,
        opType: 'Conv',
        name: node.name + '_unfused',
        outputs: [node.name + '_unfused'],
      });
      optimized.push({
        opType: 'Relu',
        inputs: [node.name + '_unfused'],
        outputs: node.outputs,
        name: node.name + '_relu',
        attributes: [],
      });
    } else if (node.opType === '_FusedMatMul') {
      optimized.push({ ...node, opType: 'MatMul', name: node.name + '_unfused' });
    } else if (node.opType === 'StopGradient') {
      continue;
    } else {
      optimized.push(node);
    }
  }
  return optimized;
}

export interface WeightRecord {
  name: string;
  dtype: string;
  data: Float32Array | Int32Array | Uint8Array;
}

export function applyQuantization(
  weights: WeightRecord[],
  targetPrecision: 'fp16' | 'int8',
): WeightRecord[] {
  return weights.map((w) => ({ ...w, dtype: targetPrecision }));
}
