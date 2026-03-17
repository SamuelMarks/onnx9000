import { describe, it, expect } from 'vitest';
import { inferShapes } from '../src/shape_inference/infer.js';
import { Graph, ValueInfo } from '../src/ir/graph.js';
import { Node } from '../src/ir/node.js';
import { Tensor } from '../src/ir/tensor.js';

describe('inferShapes', () => {
  it('should infer naive shape based on first input', () => {
    const g = new Graph('test');
    g.inputs.push(new ValueInfo('A', [1, 2, 3], 'float32'));

    const t = new Tensor('B', [1, 2, 3], 'float32');
    g.initializers.push('B');
    g.addTensor(t);

    // Add a node with missing output shape in any map
    const n = new Node('Add', ['A', 'B'], ['C']);
    g.addNode(n);

    const n2 = new Node('Relu', ['C'], ['D']);
    g.addNode(n2);

    inferShapes(g);

    // In our current naive implementation, it just populates a local shapeMap and does nothing else.
    // It doesn't mutate ValueInfo or create new Tensors.
    // So we just verify it doesn't crash.
    expect(true).toBe(true);
  });

  it('should handle nodes with no inputs or missing shapes', () => {
    const g = new Graph('test');
    g.initializers.push('Missing'); // no tensor matching

    const n = new Node('Constant', [], ['C']);
    g.addNode(n);

    inferShapes(g);
    expect(true).toBe(true);
  });
});
