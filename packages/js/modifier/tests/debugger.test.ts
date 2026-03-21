// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Graph, Node } from '@onnx9000/core';
import { GraphMutator } from '../src/GraphMutator.js';
import { GraphDebugger } from '../src/components/debugger/debugger.js';

describe('GraphDebugger (Phase 7)', () => {
  let container: HTMLElement;
  let mutator: GraphMutator;
  let graph: Graph;
  let debug: GraphDebugger;

  beforeEach(() => {
    graph = new Graph('Test');
    mutator = new GraphMutator(graph);
    container = document.createElement('div');
    debug = new GraphDebugger(container, mutator);
  });

  it('86. initSession sets up environment', async () => {
    const s = await debug.initSession();
    expect(s.initialized).toBe(true);
    expect(debug.session).toBeDefined();
  });

  it('87. setAsTemporaryOutput', () => {
    mutator.addInput('Inp1', 'float32', [1, 2]);
    const vi = mutator.graph.inputs[0]!;

    // Add output
    debug.setAsTemporaryOutput('Inp1');
    expect(mutator.graph.outputs.some((o) => o.name === 'Inp1')).toBe(true);

    // Adding again doesn't duplicate
    debug.setAsTemporaryOutput('Inp1');
    expect(mutator.graph.outputs.filter((o) => o.name === 'Inp1').length).toBe(1);

    mutator.undo();
    mutator.undo();
    expect(mutator.graph.outputs.some((o) => o.name === 'Inp1')).toBe(false);
  });

  it('88. generateDummyData creates float arrays matching shape', () => {
    mutator.addInput('InpA', 'float32', [1, 2, 2]);
    mutator.addInput('InpB', 'float32', [3]);
    const dummies = debug.generateDummyData();

    expect(dummies['InpA']).toBeInstanceOf(Float32Array);
    expect(dummies['InpA']!.length).toBe(4); // 1 * 2 * 2
    expect(dummies['InpB']!.length).toBe(3);
  });

  it('89. renderInputForm displays form for manual overriding', () => {
    mutator.addInput('InpA', 'float32', [1, 2]);
    debug.renderInputForm();
    expect(container.innerHTML).toContain('Manual Input Override');
    expect(container.innerHTML).toContain('InpA');
    expect(container.querySelectorAll('input').length).toBe(1);
  });

  it('90. 93. runGraph executes and returns profile metrics', async () => {
    mutator.addInput('A', 'float32', [1]);
    mutator.addOutput('B', 'float32', [1]);

    const { results, timeTaken } = await debug.runGraph({ A: [1.0] });
    expect(results).toHaveProperty('B');
    expect(timeTaken).toBeGreaterThanOrEqual(0);
    expect(debug.executionOutputs.has('B')).toBe(true);
  });

  it('91. renderOutputVisuals', () => {
    debug.executionOutputs.set('B', new Float32Array([1.1, 2.2]));
    debug.renderOutputVisuals();
    expect(container.innerHTML).toContain('Execution Results');
    expect(container.innerHTML).toContain('1.1');
    expect(container.innerHTML).toContain('2.2');
  });

  it('92. runSubgraph', async () => {
    const n1 = mutator.addNode('Op', ['A'], ['B']);
    const { results, timeTaken } = await debug.runSubgraph([n1.id]);
    expect(results).toHaveProperty('dummy_subgraph_out');
    expect(timeTaken).toBeGreaterThanOrEqual(0);
  });

  it('94. 95. stepNext and setBreakpoint', async () => {
    const n1 = mutator.addNode('Op', ['A'], ['B']);

    // Without session it fails
    expect(debug.stepNext()).toBeNull();

    await debug.initSession();

    // Node is there, but no breakpoint
    const r1 = debug.stepNext()!;
    expect(r1.paused).toBe(false);
    expect(r1.node.id).toBe(n1.id);

    // Set breakpoint
    debug.setBreakpoint(n1.id);
    const r2 = debug.stepNext()!;
    expect(r2.paused).toBe(true);
    expect(r2.node.id).toBe(n1.id);
  });

  it('87. setAsTemporaryOutput early returns for unknown edge', () => {
    debug.setAsTemporaryOutput('NonExistent');
    expect(mutator.graph.outputs.length).toBe(0);
  });

  it('94. 95. stepNext early returns when no nodes', async () => {
    await debug.initSession();
    expect(debug.stepNext()).toBeNull();
  });
});
