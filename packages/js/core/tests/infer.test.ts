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

    // Verify it added 'C' and 'D' to valueInfo with correct shapes
    expect(g.valueInfo.length).toBe(2);
    expect(g.valueInfo[0].name).toBe('C');
    expect(g.valueInfo[0].shape).toEqual([1, 2, 3]);
    expect(g.valueInfo[0].dtype).toBe('float32');

    expect(g.valueInfo[1].name).toBe('D');
    expect(g.valueInfo[1].shape).toEqual([1, 2, 3]);
    expect(g.valueInfo[1].dtype).toBe('float32');
  });

  it('should handle nodes with no inputs or missing shapes', () => {
    const g = new Graph('test');
    g.initializers.push('Missing'); // no tensor matching

    const n = new Node('Constant', [], ['C']);
    g.addNode(n);

    const n2 = new Node('UnknownInput', ['not_there'], ['D']);
    g.addNode(n2);

    inferShapes(g);

    // Constant has no inputs, so naive infer won't add a shape for 'C' currently
    // UnknownInput has input with no known shape, so it won't add 'D'
    expect(g.valueInfo.length).toBe(0);
  });

  it('should skip already tracked outputs', () => {
    const g = new Graph('test');
    g.inputs.push(new ValueInfo('A', [1], 'float32'));
    g.outputs.push(new ValueInfo('B', [1], 'float32'));

    const n = new Node('Identity', ['A'], ['B']);
    g.addNode(n);

    inferShapes(g);
    expect(g.valueInfo.length).toBe(0); // B is already in outputs
  });
});
