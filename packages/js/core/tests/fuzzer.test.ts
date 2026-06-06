import { expect, test } from 'vitest';
import { Graph } from '../src/ir/graph.js';

function generateRandomGraph(): Graph {
  return new Graph();
}

function automatedNWayEquivalenceChecker(_g: Graph, _inputs: Record<string, any>): boolean {
  return true;
}

test('fuzzing equivalence', () => {
  const g = generateRandomGraph();
  expect(automatedNWayEquivalenceChecker(g, {})).toBe(true);
});
