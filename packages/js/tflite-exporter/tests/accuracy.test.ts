import { describe, it, expect } from 'vitest';
import { Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import { LayoutOptimizer } from '../src/compiler/layout';

describe('Layout Modifications Accuracy', () => {
  it('should maintain numerical accuracy during NCHW to NHWC layout transpositions', () => {
    // 306. Check numerical accuracy of NCHW to NHWC layout modifications natively.
    const graph = new Graph('TestGraph');
    graph.inputs.push(new ValueInfo('X', [1, 2, 2, 2], 'float32'));

    // NCHW tensor: [1, 2, 2, 2]
    // 1 batch, 2 channels, 2 height, 2 width
    // Data:
    // C0: [[1, 2], [3, 4]]
    // C1: [[5, 6], [7, 8]]
    // Flat: 1, 2, 3, 4, 5, 6, 7, 8

    const nchwData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    graph.tensors['W'] = new Tensor('W', [1, 2, 2, 2], 'float32', true, false, nchwData);
    graph.nodes.push(new Node('Conv', ['X', 'W'], ['Y'], {}, 'conv1'));

    const optimizer = new LayoutOptimizer(graph, false);
    optimizer.optimize();

    // Weight should be transposed to NHWC: [1, 2, 2, 2] (O, H, W, I)
    // Wait, Conv weights are [O, I, H, W] -> [O, H, W, I]
    // So O=1, I=2, H=2, W=2.
    // Original:
    // O0, I0, H0, W0 = 1
    // O0, I0, H0, W1 = 2
    // O0, I0, H1, W0 = 3
    // O0, I0, H1, W1 = 4
    // O0, I1, H0, W0 = 5
    // O0, I1, H0, W1 = 6
    // O0, I1, H1, W0 = 7
    // O0, I1, H1, W1 = 8

    // Transposed to NHWC [O, H, W, I]
    // O0, H0, W0, I0 = 1
    // O0, H0, W0, I1 = 5
    // O0, H0, W1, I0 = 2
    // O0, H0, W1, I1 = 6
    // O0, H1, W0, I0 = 3
    // O0, H1, W0, I1 = 7
    // O0, H1, W1, I0 = 4
    // O0, H1, W1, I1 = 8

    const transposed = graph.tensors['W']!.data as Float32Array;
    expect(Array.from(transposed)).toEqual([1, 5, 2, 6, 3, 7, 4, 8]);
  });
});
