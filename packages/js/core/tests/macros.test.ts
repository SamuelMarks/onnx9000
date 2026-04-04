import { expect, test } from 'vitest';
import { Graph } from '../src/ir/graph.js';
import { Tensor } from '../src/ir/tensor.js';
import { irMacro, MacroExpander, MacroMatcher } from '../src/macros.js';

class TestClass {
  @irMacro('TestMacro')
  testM(x: Tensor): Tensor {
    return x;
  }
}

test('irMacro decorator', () => {
  const t = new TestClass();
  const x = new Tensor('x', [1], 1, false, false, new Float32Array());
  const out = t.testM(x);
  expect(out.name).toBe('TestMacro_out');
});

test('MacroExpander', () => {
  const expander = new MacroExpander();
  const graph = new Graph();
  const out = expander.apply(graph);
  expect(out).toBe(graph);
});

test('MacroMatcher', () => {
  const matcher = new MacroMatcher();
  const graph = new Graph();
  const out = matcher.apply(graph);
  expect(out).toBe(graph);
});
