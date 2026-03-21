import { describe, it, expect, beforeEach } from 'vitest';
import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { GraphMutator } from '../src/GraphMutator.js';
import { GraphValidator } from '../src/GraphValidator.js';

describe('GraphValidator and Advanced Mutations', () => {
  let graph: Graph;
  let mutator: GraphMutator;
  let validator: GraphValidator;

  beforeEach(() => {
    graph = new Graph('TestGraph');
    mutator = new GraphMutator(graph);
    validator = new GraphValidator(graph);
  });

  it('26, 27. Detect dangling nodes', () => {
    mutator.addNode('Op', ['A'], ['B'], {}, 'Node1'); // B is not used
    let result = validator.verify();
    expect(result.isValid).toBe(false);
    expect(result.danglingNodes).toContain('Node1');

    // Output makes it not dangling
    mutator.addOutput('B', 'float32', [1]);
    result = validator.verify();
    expect(result.isValid).toBe(false); // A is unresolved!
    expect(result.danglingNodes).toHaveLength(0);
  });

  it('28. Detect unresolved inputs', () => {
    mutator.addNode('Op', ['A'], ['B'], {}, 'Node1');
    mutator.addOutput('B', 'float32', [1]);

    const result = validator.verify();
    expect(result.isValid).toBe(false);
    expect(result.unresolvedInputs).toContain('A');
  });

  it('29. Detect cyclic dependencies', () => {
    // A -> Node1 -> B -> Node2 -> C -> Node3 -> B (cycle B -> C -> B)
    mutator.addInput('A', 'float32', [1]);
    mutator.addNode('Op1', ['A'], ['B'], {}, 'Node1');
    mutator.addNode('Op2', ['B'], ['C'], {}, 'Node2');
    mutator.addNode('Op3', ['C'], ['B'], {}, 'Node3');
    mutator.addOutput('C', 'float32', [1]);

    const result = validator.verify();
    expect(result.isValid).toBe(false);
    expect(result.cyclicDependencies).toContain('Node2');
  });

  it('30, 33. Detect type and dimension mismatches (mock)', () => {
    mutator.addInput('A', 'float32', [10, 5]);
    mutator.addInput('B', 'int32', [6, 10]); // Mismatch on dim and type
    mutator.addNode('MatMul', ['A', 'B'], ['C'], {}, 'MM');
    mutator.addOutput('C', 'float32', [10, 10]);

    const result = validator.verify();
    expect(result.isValid).toBe(false);
    expect(result.typeMismatches.length).toBeGreaterThan(0);
    expect(result.dimensionMismatches.length).toBeGreaterThan(0);
  });

  it('34. Override shape', () => {
    mutator.overrideShape('B', [10, 20], 'float32');
    expect(graph.valueInfo.length).toBe(1);
    expect(graph.valueInfo[0]!.shape).toEqual([10, 20]);
    expect(graph.valueInfo[0]!.dtype).toBe('float32');

    mutator.undo();
    expect(graph.valueInfo.length).toBe(0);
  });

  it('35. Dead Code Elimination (cleanGraph)', () => {
    mutator.addInput('A', 'float32', [1]);
    mutator.addNode('Op1', ['A'], ['B'], {}, 'Node1'); // Used
    mutator.addNode('Op2', ['B'], ['C'], {}, 'Node2'); // Used
    mutator.addNode('Op3', ['B'], ['D'], {}, 'Node3'); // Dead
    mutator.addOutput('C', 'float32', [1]);

    mutator.cleanGraph();
    expect(graph.nodes.length).toBe(2);
    expect(graph.getNode('Node3')).toBeNull();

    mutator.undo();
    expect(graph.nodes.length).toBe(3);
  });

  it('268. detects missing shapes on inputs', () => {
    graph.inputs = [{ name: 'A', dtype: 'float32', shape: [], id: '' }];
    const res = validator.verify();
    expect(res.isValid).toBe(false);
    expect(res.missingShapes).toContain('A');
  });
});
