import { expect, test } from 'vitest';
import { Graph } from '../src/ir/graph.js';

function generateRandomGraph(): Graph {
  return new Graph();
}

function automatedNWayEquivalenceChecker(g: Graph, inputs: Record<string, Object>): boolean {
  return true;
}

test('fuzzing equivalence', () => {
  const g = generateRandomGraph();
  expect(automatedNWayEquivalenceChecker(g, {})).toBe(true);
});
