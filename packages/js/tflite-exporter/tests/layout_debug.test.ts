import { describe, it } from 'vitest';
import { Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import { LayoutOptimizer } from '../src/compiler/layout';

describe('debug', () => {
  it('debug', () => {
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
    graph.nodes.push(new Node('Relu', ['Y'], ['Z'], {}, 'relu1'));

    const optimizer = new LayoutOptimizer(graph, false);
    optimizer.optimize();

    console.log(graph.nodes.map((n) => n.opType));
  });
});
