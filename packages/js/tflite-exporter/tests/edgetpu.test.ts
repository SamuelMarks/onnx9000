import { describe, it, expect } from 'vitest';
import { Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import { EdgeTPUOptimizer } from '../src/optimizations/edgetpu';

describe('TFLite Compiler - EdgeTPU', () => {
  it('should apply EdgeTPU optimizations and generate warnings', () => {
    const graph = new Graph('TestGraph');
    graph.inputs.push(new ValueInfo('X', [1, 3, 224], 'float32'));
    graph.tensors['X'] = new Tensor('X', [1, 3, 224], 'float32', false);

    graph.nodes.push(new Node('Conv', ['X'], ['Y'], {}, 'conv1d'));
    graph.nodes.push(new Node('MatMul', ['A', 'B'], ['C'], {}, 'matmul1'));
    graph.nodes.push(new Node('LeakyRelu', ['X'], ['L'], {}, 'leaky1'));
    graph.nodes.push(
      new Node('Slice', ['X', 'Starts', 'Ends', 'Axes', 'Steps'], ['S'], {}, 'slice1'),
    );
    graph.nodes.push(new Node('Softmax', ['X'], ['S2'], {}, 'soft1'));
    graph.nodes.push(new Node('Loop', ['X'], ['L1'], {}, 'loop1'));

    graph.tensors['W_conv'] = new Tensor(
      'W_conv',
      [16, 5, 3, 3],
      'float32',
      true,
      false,
      new Float32Array(16 * 5 * 9),
    );
    graph.nodes.push(new Node('Conv', ['X2', 'W_conv'], ['Y2'], {}, 'conv2'));

    const optimizer = new EdgeTPUOptimizer(graph);
    const warnings = optimizer.optimize();

    expect(warnings.length).toBeGreaterThan(0);
    expect(warnings.some((w) => w.includes('Replaced 1 1D Convolutions'))).toBe(true);
    expect(warnings.some((w) => w.includes('Expanded 1 MatMul'))).toBe(true);
    expect(warnings.some((w) => w.includes('Emulated 1 LeakyRelu'))).toBe(true);
    expect(warnings.some((w) => w.includes('Dynamic StridedSlice detected'))).toBe(true);
    expect(warnings.some((w) => w.includes('Injected Zero-Padding into 1 Convolutions'))).toBe(
      true,
    );
    expect(warnings.some((w) => w.includes('Operation Loop (loop1) breaks strict NNAPI'))).toBe(
      true,
    );
    expect(warnings.some((w) => w.includes('Rewrote 1 Softmax operations'))).toBe(true);
  });
});
