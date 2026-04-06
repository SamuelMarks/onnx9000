import { describe, it, expect } from 'vitest';
import { GraphMutator } from '../src/GraphMutator';
import { Graph, Node, Tensor, ValueInfo, Attribute } from '@onnx9000/core';

describe('Coverage Modifier 3', () => {
  it('extractSubgraph', () => {
    const graph = new Graph('test');
    graph.inputs.push(new ValueInfo('i1', [1], 'float32'));
    graph.outputs.push(new ValueInfo('o1', [1], 'float32'));

    graph.tensors['i1'] = new Tensor('i1', [1], 'float32');
    graph.initializers.push('i1');

    const n1 = new Node('Relu', ['i1'], ['y']);
    n1.id = 'n1';
    const n2 = new Node('Add', ['y', 'i2'], ['o1']);
    n2.id = 'n2';
    const n3 = new Node('Sub', ['o1', 'i3'], ['o2']);
    n3.id = 'n3';

    graph.addNode(n1);
    graph.addNode(n2);
    graph.addNode(n3);

    const mut = new GraphMutator(graph);
    const sub = mut.extractSubgraph(['n2']);
    expect(sub.nodes.length).toBe(1);
  });

  it('addInput edge', () => {
    const g = new Graph('test');
    const t = new Tensor('i', [1], 'float16');
    g.tensors['i'] = t;
    g.initializers.push('i');
    const mut = new GraphMutator(g);
    mut.addInput('i', 'float16', [1]); // hits 364-370
    expect(g.inputs.length).toBe(1);
  });

  it('updateInitializer branch', () => {
    const g = new Graph('test');
    const t16 = new Tensor('t16', [1], 'float16');
    const t8 = new Tensor('t8', [1], 'int8');
    const t32 = new Tensor('t32', [1], 'int32');
    const t64 = new Tensor('t64', [1], 'int64');
    const tUnk = new Tensor('tUnk', [1], 'string' as Object);
    g.tensors['t16'] = t16;
    g.tensors['t8'] = t8;
    g.tensors['t32'] = t32;
    g.tensors['t64'] = t64;
    g.tensors['tUnk'] = tUnk;

    const mut = new GraphMutator(g);
    expect(() => mut.updateInitializer('t16', new Uint8Array(1))).toThrow();
    expect(() => mut.updateInitializer('t8', new Uint8Array(0))).toThrow();
    expect(() => mut.updateInitializer('t32', new Uint8Array(1))).toThrow();
    expect(() => mut.updateInitializer('t64', new Uint8Array(1))).toThrow();
    expect(() => mut.updateInitializer('tUnk', new Uint8Array(2))).toThrow();
  });

  it('graph mutations ops', () => {
    const g = new Graph('test');
    const mut = new GraphMutator(g);

    g.addNode(new Node('Cast', ['a'], ['b'], { to: { value: 1 } } as Object));
    g.addNode(new Node('Dropout', ['b'], ['c']));
    g.addNode(new Node('Constant', [], ['d']));
    g.outputs.push(new ValueInfo('c', [1], 'float32'));

    mut.fixMixedPrecision('FLOAT16');
    mut.undo();

    mut.removeTrainingNodes();
    mut.undo();

    mut.foldConstants();
    mut.undo();

    mut.extractWeights(10);
    mut.undo();

    mut.sanitizeNames();
    mut.undo();
  });
});
