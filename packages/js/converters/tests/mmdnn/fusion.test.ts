import { describe, it, expect } from 'vitest';
import { NodeFusionRegistry } from '../src/mmdnn/fusion.js';
import { Graph } from '@onnx9000/core';

describe('fusion', () => {
  it('should fuse nodes', () => {
    const reg = new NodeFusionRegistry();
    const g: any = {
      nodes: [
        { opType: 'Conv', name: 'c1', outputs: ['o1'], inputs: [] },
        { opType: 'BatchNormalization', name: 'bn1', inputs: ['o1'], outputs: ['o2'] },
      ],
    };
    const rep: any = { info: () => {} };

    const newG = reg.applyFusions(g, rep);
    expect(newG.nodes.length).toBe(1);
    expect(newG.nodes[0].opType).toBe('Conv');
    expect(newG.nodes[0].outputs[0]).toBe('o2');
  });
});
