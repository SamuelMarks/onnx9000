import { describe, it, expect } from 'vitest';
import { Graph, Node, Tensor, ValueInfo, Attribute } from '@onnx9000/core';
import { LayoutOptimizer } from '../src/compiler/layout';

describe('LayoutOptimizer - Edge Cases', () => {
  it('should strip Identity and Dropout ops', () => {
    const graph = new Graph('TestGraph');
    graph.inputs.push(new ValueInfo('X', [1, 3], 'float32'));

    graph.nodes.push(new Node('Identity', ['X'], ['Y'], {}, 'id1'));
    graph.nodes.push(new Node('Dropout', ['Y'], ['Z'], {}, 'drop1'));
    graph.nodes.push(new Node('Relu', ['Z'], ['Out'], {}, 'relu1'));

    graph.outputs.push(new ValueInfo('Out', [1, 3], 'float32'));

    const optimizer = new LayoutOptimizer(graph, false);
    optimizer.optimize();

    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0]!.opType).toBe('Relu');
    expect(graph.nodes[0]!.inputs[0]).toBe('X');
  });

  it('should rewrite negative axes correctly', () => {
    const graph = new Graph('TestGraph');
    graph.inputs.push(new ValueInfo('X', [1, 3, 10, 10], 'float32'));

    graph.nodes.push(
      new Node(
        'Concat',
        ['X', 'X'],
        ['Y'],
        {
          axis: new Attribute('axis', 'INT', -1),
        },
        'concat1',
      ),
    );

    graph.nodes.push(
      new Node(
        'Squeeze',
        ['Y'],
        ['Z'],
        {
          axes: new Attribute('axes', 'INTS', [-2, -1]),
        },
        'squeeze1',
      ),
    );

    const optimizer = new LayoutOptimizer(graph, true); // keep nchw to just test this pass
    optimizer.optimize();

    expect(graph.nodes[0]!.attributes['axis']!.value).toBe(3); // -1 + 4
    expect(graph.nodes[1]!.attributes['axes']!.value).toEqual([2, 3]); // -2+4, -1+4
  });

  it('should decompose BatchNormalization', () => {
    const graph = new Graph('TestGraph');
    graph.inputs.push(new ValueInfo('X', [1, 3, 10, 10], 'float32'));
    graph.tensors['Scale'] = new Tensor(
      'Scale',
      [3],
      'float32',
      true,
      false,
      new Float32Array([1, 1, 1]),
    );
    graph.tensors['B'] = new Tensor('B', [3], 'float32', true, false, new Float32Array([0, 0, 0]));
    graph.tensors['Mean'] = new Tensor(
      'Mean',
      [3],
      'float32',
      true,
      false,
      new Float32Array([1, 2, 3]),
    );
    graph.tensors['Var'] = new Tensor(
      'Var',
      [3],
      'float32',
      true,
      false,
      new Float32Array([0.1, 0.1, 0.1]),
    );

    graph.nodes.push(
      new Node(
        'BatchNormalization',
        ['X', 'Scale', 'B', 'Mean', 'Var'],
        ['Y'],
        {
          epsilon: new Attribute('epsilon', 'FLOAT', 1e-5),
        },
        'bn1',
      ),
    );

    const optimizer = new LayoutOptimizer(graph, true);
    optimizer.optimize();

    expect(graph.nodes.length).toBe(2);
    expect(graph.nodes[0]!.opType).toBe('Mul');
    expect(graph.nodes[1]!.opType).toBe('Add');
  });

  it('should fuse BatchNormalization into Conv', () => {
    const graph = new Graph('TestGraph');
    graph.inputs.push(new ValueInfo('X', [1, 3, 10, 10], 'float32'));

    graph.tensors['W'] = new Tensor(
      'W',
      [3, 3, 3, 3],
      'float32',
      true,
      false,
      new Float32Array(81),
    );
    graph.tensors['Scale'] = new Tensor(
      'Scale',
      [3],
      'float32',
      true,
      false,
      new Float32Array([1, 1, 1]),
    );
    graph.tensors['B'] = new Tensor('B', [3], 'float32', true, false, new Float32Array([0, 0, 0]));
    graph.tensors['Mean'] = new Tensor(
      'Mean',
      [3],
      'float32',
      true,
      false,
      new Float32Array([1, 2, 3]),
    );
    graph.tensors['Var'] = new Tensor(
      'Var',
      [3],
      'float32',
      true,
      false,
      new Float32Array([0.1, 0.1, 0.1]),
    );

    graph.nodes.push(new Node('Conv', ['X', 'W'], ['Y'], {}, 'conv1'));
    graph.nodes.push(
      new Node(
        'BatchNormalization',
        ['Y', 'Scale', 'B', 'Mean', 'Var'],
        ['Z'],
        {
          epsilon: new Attribute('epsilon', 'FLOAT', 1e-5),
        },
        'bn1',
      ),
    );

    const optimizer = new LayoutOptimizer(graph, true);
    optimizer.optimize();

    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0]!.opType).toBe('Conv');
    expect(graph.nodes[0]!.inputs.length).toBe(3); // W and fused B
  });
});
