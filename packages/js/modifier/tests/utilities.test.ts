import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Graph } from '@onnx9000/core';
import { GraphMutator } from '../src/GraphMutator.js';
import { ModifierUtilities } from '../src/components/utilities.js';

describe('ModifierUtilities', () => {
  let graph: Graph;
  let mutator: GraphMutator;
  let utils: ModifierUtilities;

  beforeEach(() => {
    graph = new Graph('Test');
    mutator = new GraphMutator(graph);
    utils = new ModifierUtilities(mutator);
  });

  it('extractSubgraph handles tensors gracefully', () => {
    mutator.addInitializer('W1', 'float32', [1], new Float32Array([1]));
    mutator.addInput('A', 'float32', [1]);
    const n1 = mutator.addNode('MatMul', ['A', 'W1'], ['B'], {}, 'Node1');
    const sub = utils.extractSubgraph([n1.id]);
    expect(sub.tensors['W1']).toBeDefined();
  });

  it('69. extractSubgraph grabs nodes and figures out borders', () => {
    mutator.addInput('A', 'float32', [1]);
    const n1 = mutator.addNode('Op1', ['A'], ['B'], {}, 'Node1');
    const n2 = mutator.addNode('Op2', ['B'], ['C'], {}, 'Node2');
    mutator.addNode('Op3', ['C'], ['D'], {}, 'Node3');
    mutator.graph.valueInfo.push({ name: 'B', shape: [1], dtype: 'float32', id: '' });
    mutator.graph.valueInfo.push({ name: 'C', shape: [1], dtype: 'float32', id: '' });
    mutator.graph.outputs.push({ name: 'D', shape: [1], dtype: 'float32', id: '' });

    const sub = utils.extractSubgraph([n1.id, n2.id]);
    expect(sub.nodes.length).toBe(2);
    expect(sub.inputs.some((i) => i.name === 'A')).toBe(true);
    expect(sub.outputs.some((o) => o.name === 'C')).toBe(true);
  });

  it('71. changeOpsetVersion modifies or appends', () => {
    graph.opsetImports['ai.onnx'] = 12;
    utils.changeOpsetVersion('ai.onnx', 14);
    expect(graph.opsetImports['ai.onnx']).toBe(14);

    utils.changeOpsetVersion('com.microsoft', 1);
    expect(graph.opsetImports['com.microsoft']).toBe(1);

    utils.changeOpsetVersion('ai.onnx', 15);
    expect(graph.opsetImports['ai.onnx']).toBe(15);
  });

  it('73. injectCastNode inserts cast', () => {
    const newName = utils.injectCastNode('Edge1', 'FLOAT');
    expect(newName).toBe('Edge1_casted');
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0]!.opType).toBe('Cast');
    expect(graph.nodes[0]!.attributes['to']!.value).toBe(1);

    utils.injectCastNode('Edge2', 'INT8');
    expect(graph.nodes[1]!.attributes['to']!.value).toBe(7);
  });

  it('72. regexRenameNodes fallbacks to id', () => {
    const node = mutator.addNode('Op', [], [], {});
    node.name = '';
    utils.regexRenameNodes(node.id, 'block_1');
    expect(node.name).toBe('block_1');
  });

  it('66. changeBatchSize updates dimension 0 globally', () => {
    mutator.addInput('A', 'float32', [1, 3, 224, 224]);
    mutator.addOutput('B', 'float32', [1, 1000]);
    mutator.overrideShape('Intermediate', [1, 64, 112, 112], 'float32');

    utils.changeBatchSize(8);

    expect(graph.inputs[0]!.shape[0]).toBe(8);
    expect(graph.outputs[0]!.shape[0]).toBe(8);
    expect(graph.valueInfo[0]!.shape[0]).toBe(8);

    mutator.undo(); // undo override of Intermediate
    mutator.undo(); // undo override of B
    mutator.undo(); // undo override of A
    expect(graph.inputs[0]!.shape[0]).toBe(1);
  });

  it('67. makeDynamic sets batch size to dynamic variable', () => {
    mutator.addInput('A', 'float32', [1, 3, 224, 224]);
    utils.makeDynamic();
    expect(graph.inputs[0]!.shape[0]).toBe('batch_size');
  });

  it('68. stripInitializers removes all weights', () => {
    mutator.addInitializer('W1', 'float32', [10], new Float32Array(10));
    mutator.addInitializer('W2', 'float32', [20], new Float32Array(20));

    expect(graph.initializers.length).toBe(2);
    utils.stripInitializers();
    expect(graph.initializers.length).toBe(0);

    mutator.undo(); // W2
    mutator.undo(); // W1
    expect(graph.initializers.length).toBe(2);
  });

  it('70. insertIdentity drops node and reroutes consumers', () => {
    mutator.addNode('Op1', ['A'], ['B'], {}, 'Node1');
    mutator.addNode('Op2', ['B'], ['C'], {}, 'Node2');
    mutator.addNode('Op3', ['B'], ['D'], {}, 'Node3');

    // B is consumed by Node2 and Node3. Insert identity.
    utils.insertIdentity('B');

    // We added Node1, Node2, Node3, and now Identity
    expect(graph.nodes.length).toBe(4);

    const idNode = graph.nodes.find((n) => n.opType === 'Identity')!;
    expect(idNode.inputs[0]).toBe('B');
    expect(idNode.outputs[0]).toBe('B_identity');

    const node2 = graph.nodes.find((n) => n.name === 'Node2')!;
    const node3 = graph.nodes.find((n) => n.name === 'Node3')!;
    expect(node2.inputs[0]).toBe('B_identity');
    expect(node3.inputs[0]).toBe('B_identity');

    mutator.undo(); // reroute node3
    mutator.undo(); // reroute node2
    mutator.undo(); // node insertion

    const n2_undone = graph.nodes.find((n) => n.name === 'Node2')!;
    expect(n2_undone.inputs[0]).toBe('B');
  });

  it('70. insertIdentity early returns if no consumers', () => {
    utils.insertIdentity('NonExistent');
    expect(graph.nodes.length).toBe(0);
  });

  it('72. regexRenameNodes renames based on pattern', () => {
    mutator.addNode('Op', [], [], {}, 'layer_1/conv');
    mutator.addNode('Op', [], [], {}, 'layer_1/relu');
    mutator.addNode('Op', [], [], {}, 'layer_2/conv');

    utils.regexRenameNodes('layer_1/(.*)', 'block_1/$1');

    expect(graph.nodes[0]!.name).toBe('block_1/conv');
    expect(graph.nodes[1]!.name).toBe('block_1/relu');
    expect(graph.nodes[2]!.name).toBe('layer_2/conv'); // untouched
  });
});
