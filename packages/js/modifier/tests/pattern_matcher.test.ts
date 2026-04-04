import { expect, test } from 'vitest';
import { Graph, Node } from '@onnx9000/core';
import {
  Pattern,
  PatternMatcherEngine,
  applyAlgebraicReuse,
  applyFusionReuse,
  applyHardwareLowering,
  matches,
} from '../src/pattern_matcher.js';

test('PatternMatcherEngine', () => {
  const engine = new PatternMatcherEngine();
  engine.addRule(new Pattern('Add'), (n) => null);

  const g = new Graph();
  g.nodes.push(new Node('Add', [], []));
  g.nodes.push(new Node('Sub', [], []));

  const out = engine.apply(g);
  expect(out).toBe(g);
});

test('applyAlgebraicReuse', () => {
  const g = new Graph();
  const out = applyAlgebraicReuse(g);
  expect(out).toBe(g);
});

test('applyFusionReuse', () => {
  const g = new Graph();
  const out = applyFusionReuse(g);
  expect(out).toBe(g);
});

test('applyHardwareLowering', () => {
  const g = new Graph();
  const out = applyHardwareLowering(g);
  expect(out).toBe(g);
});

test('matches with non-empty pattern inputs', () => {
  const node = new Node('Add', [], []);
  const pattern = new Pattern('Add', ['input1']);
  expect(matches(node, pattern)).toBe(true);
});
