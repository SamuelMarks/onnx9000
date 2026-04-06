import { describe, it, expect } from 'vitest';
import { Graph, Tensor } from '@onnx9000/core';
import {
  applyRecipe,
  MagnitudePruningModifier,
  ConstantPruningModifier,
  parseRecipe,
} from '../src/sparse/modifier';

describe('Sparse Modifier', () => {
  it('Parses and applies recipes directly', () => {
    const graph = new Graph();

    const t1 = new Tensor(
      'weight1',
      [4],
      'float32',
      false,
      true,
      new Float32Array([0.1, 0.9, -0.2, 0.5]),
    );
    const t2 = new Tensor(
      'weight2',
      [4],
      'float32',
      false,
      true,
      new Float32Array([0.0, 1.0, 2.0, 3.0]),
    );
    const t3 = new Tensor('ignored', [4], 'float32', false, true, new Float32Array([1, 1, 1, 1]));
    const t4 = new Tensor('no_data', [4], 'float32', false, true, null as Object); // no data
    const t5 = new Tensor('empty_data', [0], 'float32', false, true, new Float32Array([])); // empty data

    // Let's force isInitializer!
    t1.isInitializer = true;
    t2.isInitializer = true;
    t3.isInitializer = true;
    t4.isInitializer = true;
    t5.isInitializer = true;

    graph.tensors = {
      weight1: t1,
      weight2: t2,
      ignored: t3,
      no_data: t4,
      empty_data: t5,
    };

    const mod1 = new MagnitudePruningModifier({
      params: ['re:weight.*'],
      final_sparsity: 0.5,
      leave_unmasked: ['weight2'],
    });
    mod1.apply(graph);

    const d1 = Array.from(t1.data as Float32Array);
    expect(Math.abs(d1[0])).toBeLessThan(1e-5);
    expect(Math.abs(d1[2])).toBeLessThan(1e-5);
    expect(Math.abs(d1[1] - 0.9)).toBeLessThan(1e-5);

    const mod2 = new ConstantPruningModifier({ params: ['re:weight2'], threshold: 0 });
    mod2.apply(graph);
    const d2 = Array.from(t2.data as Float32Array);
    expect(d2[0]).toBe(0);
    expect(d2[1]).toBe(1);

    const mod3 = new MagnitudePruningModifier({ params: ['re:.*'], final_sparsity: 0 });
    mod3.apply(graph);

    const mod4 = new ConstantPruningModifier({ params: ['re:no_data', 're:empty_data'] });
    mod4.apply(graph);

    const yaml = `
# comment
- !MagnitudePruningModifier
  params: ['re:weight.*']
  leave_unmasked: ['weight2']
  final_sparsity: 0.5
  
- !ConstantPruningModifier
  params: ['re:weight2']
  threshold: 0
  
- !UnknownModifier
  key: value
`;
    applyRecipe(graph, yaml);
    expect(graph.metadataProps['onnx9000_sparse_recipe']).toBe(yaml);
  });
});

describe('Sparse Modifier Defaults', () => {
  it('applies with defaults', () => {
    const graph = new Graph();
    const mod1 = new MagnitudePruningModifier({});
    mod1.apply(graph);
    const mod2 = new ConstantPruningModifier({});
    mod2.apply(graph);
  });
});
