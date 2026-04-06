import { describe, it, expect } from 'vitest';
import { Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import { LayoutOptimizer } from '../src/compiler/layout';

describe('LayoutOptimizer', () => {
  it('should keep NCHW if requested', () => {
    const graph = new Graph('TestGraph');
    const optimizer = new LayoutOptimizer(graph, true);
    optimizer.optimize();
    expect(graph.nodes.length).toBe(0);
  });

  it('should inject transposes for spatial ops', () => {
    const graph = new Graph('TestGraph');
    graph.inputs.push(new ValueInfo('X', [1, 3, 224, 224], 'float32'));
    graph.tensors['W'] = new Tensor(
      'W',
      [64, 3, 3, 3],
      'float32',
      true,
      false,
      new Float32Array(64 * 27),
    );
    graph.nodes.push(new Node('Conv', ['X', 'W'], ['Y'], {}, 'conv1'));

    const optimizer = new LayoutOptimizer(graph, false);
    optimizer.optimize();

    expect(graph.nodes.length).toBe(3);
    expect(graph.nodes[0]!.opType).toBe('Transpose');
    expect(graph.nodes[0]!.attributes['perm']?.value).toEqual([0, 2, 3, 1]);
    expect(graph.nodes[1]!.opType).toBe('Conv');
    expect(graph.nodes[2]!.opType).toBe('Transpose');
    expect(graph.nodes[2]!.attributes['perm']?.value).toEqual([0, 3, 1, 2]);

    // Check folded weight
    const w = graph.tensors['W']!;
    expect(w.shape).toEqual([64, 3, 3, 3]);
  });

  it('should push down transposes through elementwise ops', () => {
    const graph = new Graph('TestGraph');
    graph.inputs.push(new ValueInfo('X', [1, 3, 224, 224], 'float32'));
    graph.tensors['W'] = new Tensor(
      'W',
      [64, 3, 3, 3],
      'float32',
      true,
      false,
      new Float32Array(64 * 27),
    );

    graph.nodes.push(new Node('Conv', ['X', 'W'], ['Y'], {}, 'conv1'));
    // Elementwise Op right after Conv
    graph.nodes.push(new Node('Relu', ['Y'], ['Z'], {}, 'relu1'));

    const optimizer = new LayoutOptimizer(graph, false);
    optimizer.optimize();

    // Original: X -> TransIn -> Conv -> TransOut -> Relu
    // PushDown: X -> TransIn -> Conv -> Relu -> TransOut
    // Relu is now fused into Conv! So there is only 3 nodes: TransIn, Conv, TransOut
    expect(graph.nodes.length).toBe(3);
    expect(graph.nodes[0]!.opType).toBe('Transpose'); // In
    expect(graph.nodes[1]!.opType).toBe('Conv');
    expect(graph.nodes[2]!.opType).toBe('Transpose'); // Out
  });

  it('should cancel adjacent transposes', () => {
    const graph = new Graph('TestGraph');
    graph.inputs.push(new ValueInfo('X', [1, 3, 224, 224], 'float32'));

    // Two spatial ops in a row
    graph.tensors['W1'] = new Tensor(
      'W1',
      [64, 3, 3, 3],
      'float32',
      true,
      false,
      new Float32Array(64 * 27),
    );
    graph.tensors['W2'] = new Tensor(
      'W2',
      [64, 64, 3, 3],
      'float32',
      true,
      false,
      new Float32Array(64 * 64 * 9),
    );

    graph.nodes.push(new Node('Conv', ['X', 'W1'], ['Y'], {}, 'conv1'));
    graph.nodes.push(new Node('Conv', ['Y', 'W2'], ['Z'], {}, 'conv2'));

    const optimizer = new LayoutOptimizer(graph, false);
    optimizer.optimize();

    // Original: X -> TransIn1 -> Conv1 -> TransOut1 -> TransIn2 -> Conv2 -> TransOut2
    // Cancelled: X -> TransIn1 -> Conv1 -> Conv2 -> TransOut2
    expect(graph.nodes.length).toBe(4);
    expect(graph.nodes[0]!.opType).toBe('Transpose'); // In
    expect(graph.nodes[1]!.opType).toBe('Conv'); // Conv1
    expect(graph.nodes[2]!.opType).toBe('Conv'); // Conv2
    expect(graph.nodes[3]!.opType).toBe('Transpose'); // Out
  });

  it('should fold transposed constants properly (Gemm, Depthwise, ConvTranspose)', () => {
    const graph = new Graph('TestGraph');

    // Depthwise Conv
    graph.tensors['W_dw'] = new Tensor(
      'W_dw',
      [3, 1, 3, 3],
      'float32',
      true,
      false,
      new Float32Array(27),
    );
    graph.nodes.push(
      new Node('Conv', ['X1', 'W_dw'], ['Y1'], { group: { value: 3 } as Object }, 'conv_dw'),
    );

    // ConvTranspose
    graph.tensors['W_ct'] = new Tensor(
      'W_ct',
      [3, 64, 3, 3],
      'float32',
      true,
      false,
      new Float32Array(3 * 64 * 9),
    );
    graph.nodes.push(new Node('ConvTranspose', ['X2', 'W_ct'], ['Y2'], {}, 'conv_t'));

    // Gemm (transB = 0)
    graph.tensors['W_gemm'] = new Tensor(
      'W_gemm',
      [10, 20],
      'float32',
      true,
      false,
      new Float32Array(200),
    );
    graph.nodes.push(new Node('Gemm', ['X3', 'W_gemm'], ['Y3'], {}, 'gemm1'));

    const optimizer = new LayoutOptimizer(graph, false);
    optimizer.optimize();

    expect(graph.tensors['W_dw']!.shape).toEqual([1, 3, 3, 3]);
    expect(graph.tensors['W_ct']!.shape).toEqual([64, 3, 3, 3]);
    expect(graph.tensors['W_gemm']!.shape).toEqual([20, 10]);
  });
});
