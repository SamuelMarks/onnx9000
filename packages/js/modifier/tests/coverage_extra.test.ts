import { describe, it, expect, beforeEach } from 'vitest';
import { Graph, Node, Attribute, ValueInfo, Tensor } from '@onnx9000/core';
import { GraphMutator } from '../src/GraphMutator.js';
import { GraphValidator } from '../src/GraphValidator.js';
import { DagreLayoutEngine } from '../src/render/layout.js';

describe('Extra coverage for GraphMutator', () => {
  let graph: Graph;
  let mutator: GraphMutator;

  beforeEach(() => {
    graph = new Graph('TestGraph');
    mutator = new GraphMutator(graph);
  });

  it('undo addInitializer and updateInitializer', () => {
    const data1 = new Float32Array([1, 2]);
    mutator.addInitializer('W', 'float32', [2], data1);
    mutator.undo(); // undo add
    expect(graph.initializers.includes('W')).toBe(false);
    expect(graph.tensors['W']).toBeUndefined();
    mutator.redo(); // redo add

    // add again when it's already there
    mutator.addInitializer('W', 'float32', [2], data1);
  });

  it('undo convertInputToInitializer', () => {
    mutator.addInput('Const', 'float32', [1]);
    const data = new Float32Array([42]);
    mutator.convertInputToInitializer('Const', data);
    mutator.undo(); // undo convertInputToInitializer
    expect(graph.inputs.length).toBe(1);
    expect(graph.initializers.includes('Const')).toBe(false);
  });

  it('setNodeAttribute overwrite existing', () => {
    mutator.addNode('Conv', ['X'], ['Y'], {}, 'C1');
    mutator.setNodeAttribute('C1', 'kernel_shape', [3, 3], 'INTS');
    mutator.setNodeAttribute('C1', 'kernel_shape', [5, 5], 'INTS');
    expect(graph.getNode('C1')!.attributes['kernel_shape']!.value).toEqual([5, 5]);
    mutator.undo(); // undo overwrite
    expect(graph.getNode('C1')!.attributes['kernel_shape']!.value).toEqual([3, 3]);
    mutator.undo(); // undo set new
    expect(graph.getNode('C1')!.attributes['kernel_shape']).toBeUndefined();
  });

  it('cyclic dependency in topologicalSort', () => {
    mutator.addNode('Op1', ['B'], ['A'], {}, 'N1');
    mutator.addNode('Op2', ['A'], ['B'], {}, 'N2');
    // Calling topological sort with a cycle should skip and not crash
    mutator.topologicalSort();
    // N1 or N2 will just be arbitrarily sorted or one skipped in processing
    expect(graph.nodes.length).toBe(2);
  });

  it('inferShapesGlobally', async () => {
    mutator.addInput('A', 'float32', [1, 2]);
    mutator.addNode('Relu', ['A'], ['B'], {}, 'R');

    mutator.inferShapesGlobally();
    // Since it uses dynamic import, we must wait a bit for it to resolve
    await new Promise((r) => setTimeout(r, 100));
    expect(graph.valueInfo.some((v) => v.name === 'B')).toBe(true);
  });

  it('overrideShape overwrite existing', () => {
    mutator.overrideShape('T', [1], 'float32');
    mutator.overrideShape('T', [2], 'float32');
    expect(graph.valueInfo.find((v) => v.name === 'T')!.shape).toEqual([2]);
    mutator.undo();
    expect(graph.valueInfo.find((v) => v.name === 'T')!.shape).toEqual([1]);
    mutator.undo();
    expect(graph.valueInfo.find((v) => v.name === 'T')).toBeUndefined();
  });

  it('early returns when things do not exist', () => {
    // Should safely return void without crashing
    mutator.removeNode('nonexistent');
    mutator.renameNode('nonexistent', 'new');
    mutator.replaceNode('nonexistent', new Node('Id', [], []));
    mutator.changeNodeOpType('nonexistent', 'Id');
    mutator.removeInput('nonexistent');
    mutator.removeOutput('nonexistent');
    mutator.removeInitializer('nonexistent');
    mutator.updateInitializer('nonexistent', new Float32Array());
    mutator.convertInputToInitializer('nonexistent', new Float32Array());
    mutator.convertInitializerToInput('nonexistent');
    mutator.setNodeAttribute('nonexistent', 'a', 1, 'INT');
    mutator.removeNodeAttribute('nonexistent', 'a');

    mutator.addNode('Id', ['A'], ['B'], {}, 'N1');
    mutator.removeNodeAttribute('N1', 'nonexistent_attr');
  });

  it('cleanGraph returns early when nothing to remove', () => {
    mutator.addInput('A', 'float32', [1]);
    mutator.addNode('Id', ['A'], ['B'], {}, 'N1');
    mutator.addOutput('B', 'float32', [1]);
    mutator.cleanGraph(); // All nodes needed, nothing removed
    expect(graph.nodes.length).toBe(1);
  });

  it('renameInput globally when not a graph input/output', () => {
    mutator.addNode('Id', ['A'], ['B'], {}, 'N1');
    // Renaming 'A' to 'A_new' when it's not in graph.inputs
    mutator.renameInput('A', 'A_new');
    expect(graph.nodes[0]!.inputs[0]).toBe('A_new');
  });
});

describe('Extra coverage for GraphValidator', () => {
  let graph: Graph;
  let mutator: GraphMutator;
  let validator: GraphValidator;

  beforeEach(() => {
    graph = new Graph('TestGraph');
    mutator = new GraphMutator(graph);
    validator = new GraphValidator(graph);
  });

  it('validates dim mismatches specifically', () => {
    // Only dimension mismatch
    mutator.addInput('A', 'float32', [10, 5]);
    mutator.addInput('B', 'float32', [6, 10]); // Mismatch on dim but same type
    mutator.addNode('MatMul', ['A', 'B'], ['C'], {}, 'MM');
    mutator.addOutput('C', 'float32', [10, 10]);

    const result = validator.verify();
    expect(result.isValid).toBe(false);
    expect(result.typeMismatches.length).toBe(0); // false for type
    expect(result.dimensionMismatches.length).toBeGreaterThan(0); // true for dim
  });

  it('node without name fallbacks to id', () => {
    const node = mutator.addNode('Op', ['A'], ['B']); // no name
    node.name = ''; // force empty
    mutator.addNode('Op2', ['B'], ['A']); // creates cycle
    graph.nodes[1]!.name = '';

    const result = validator.verify();
    expect(result.isValid).toBe(false);
    expect(result.cyclicDependencies).toContain(graph.nodes[1]!.id);
  });

  it('visited twice without cycle', () => {
    // A -> B -> D
    // A -> C -> D
    mutator.addNode('Id', ['A'], ['B'], {}, 'N1');
    mutator.addNode('Id', ['A'], ['C'], {}, 'N2');
    mutator.addNode('Add', ['B', 'C'], ['D'], {}, 'N3');
    mutator.addInput('A', 'float32', [1]);
    mutator.addOutput('D', 'float32', [1]);

    const result = validator.verify();
    expect(result.isValid).toBe(true);
  });

  it('graph with initializers, tensors and valueInfo', () => {
    mutator.addInitializer('W', 'float32', [1], new Float32Array([1]));
    mutator.overrideShape('B', [1], 'float32'); // creates valueInfo
    mutator.addInput('A', 'float32', [1]);
    mutator.addNode('MatMul', ['A', 'W'], ['B'], {}, 'N1');
    mutator.addOutput('B', 'float32', [1]);

    const result = validator.verify();
    // It will check type and dimension mismatches
    expect(result.isValid).toBe(true);
  });
});

describe('More Edge case coverage', () => {
  let graph: Graph;
  let mutator: GraphMutator;

  beforeEach(() => {
    graph = new Graph('TestGraph');
    mutator = new GraphMutator(graph);
  });

  it('undo rename of graph input and output', () => {
    mutator.addInput('A', 'float32', [1]);
    mutator.addOutput('C', 'float32', [1]);
    mutator.addNode('Op', ['A'], ['C'], {}, 'N1');

    mutator.renameInput('A', 'A_new');
    mutator.undo();
    expect(graph.inputs[0]!.name).toBe('A');

    mutator.renameOutput('C', 'C_new');
    mutator.undo();
    expect(graph.outputs[0]!.name).toBe('C');
  });
});

describe('Extra GraphValidator branches', () => {
  let graph: Graph;
  let mutator: GraphMutator;
  let validator: GraphValidator;

  beforeEach(() => {
    graph = new Graph('TestGraph');
    mutator = new GraphMutator(graph);
    validator = new GraphValidator(graph);
  });

  it('dangling node with valid name', () => {
    mutator.addNode('Op', ['A'], ['B'], {}, 'HasName');
    const result = validator.verify();
    expect(result.isValid).toBe(false);
    expect(result.danglingNodes).toContain('HasName');
  });
});

describe('Extra GraphValidator branches 2', () => {
  let graph: Graph;
  let mutator: GraphMutator;
  let validator: GraphValidator;

  beforeEach(() => {
    graph = new Graph('TestGraph');
    mutator = new GraphMutator(graph);
    validator = new GraphValidator(graph);
  });

  it('node with no outputs', () => {
    mutator.addNode('NoOut', ['A'], []);
    const result = validator.verify();
    expect(result.isValid).toBe(false);
  });
});

describe('Dagre coverage', () => {
  it('gracefully handles missing producers', () => {
    const graph = new Graph('TestGraph');
    const engine = new DagreLayoutEngine('TB');
    graph.nodes.push(new Node('Op', ['A'], ['B']));
    const layout = engine.compute(graph);
    expect(layout.edges.length).toBe(0);
  });
});

describe('Extra mutate coverage', () => {
  let graph: Graph;
  let mutator: GraphMutator;

  beforeEach(() => {
    graph = new Graph('TestGraph');
    mutator = new GraphMutator(graph);
  });

  it('handles removeInput when it does not exist', () => {
    mutator.removeInput('nonexistent');
    expect(graph.inputs.length).toBe(0);
  });

  it('handles removeOutput when it does not exist', () => {
    mutator.removeOutput('nonexistent');
    expect(graph.outputs.length).toBe(0);
  });
});

describe('Dagre coverage missing node', () => {
  it('gracefully handles missing nodes when getting layer', () => {
    const engine = new DagreLayoutEngine('TB');
    const graph = new Graph('TestGraph');
    const n = new Node('A', [], []);
    graph.nodes.push(n);
    const orig = graph.nodes.find;
    graph.nodes.find = () => undefined;
    const layout = engine.compute(graph);
    expect(layout.nodes.size).toBe(0);
    graph.nodes.find = orig;
  });
});

describe('Extra GraphMutator 221, 249', () => {
  let graph: Graph;
  let mutator: GraphMutator;

  beforeEach(() => {
    graph = new Graph('TestGraph');
    mutator = new GraphMutator(graph);
  });

  it('undoes addInput and addOutput fully', () => {
    mutator.addInput('IN', 'float32', [1]);
    mutator.undo();
    expect(graph.inputs.length).toBe(0);
    mutator.addOutput('OUT', 'float32', [1]);
    mutator.undo();
    expect(graph.outputs.length).toBe(0);
  });
});

describe('Extra GraphValidator branches 3', () => {
  let graph: Graph;
  let mutator: GraphMutator;
  let validator: GraphValidator;

  beforeEach(() => {
    graph = new Graph('TestGraph');
    mutator = new GraphMutator(graph);
    validator = new GraphValidator(graph);
  });

  it('node name exists but is empty string so falls back to id', () => {
    const node = mutator.addNode('NoOut', ['A'], ['B']);
    node.name = '';
    const result = validator.verify();
    expect(result.isValid).toBe(false);
    expect(result.danglingNodes).toContain(node.id);
  });
});
