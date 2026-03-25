import { describe, it, expect } from 'vitest';
import { LightGBMParser } from '../../../src/mmdnn/lightgbm/parser.js';

describe('LightGBMParser', () => {
  it('should parse lightgbm tree content to TreeEnsembleRegressor', () => {
    const parser = new LightGBMParser();
    const graph = parser.parseModel('tree_info=... \n some other tree info');
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('TreeEnsembleRegressor');
    expect(graph.nodes[0].attributes['nodes_treeids']).toBeDefined();
    expect(graph.nodes[0].attributes['n_targets']).toBeDefined();
  });

  it('should fallback to Identity when tree not found', () => {
    const parser = new LightGBMParser();
    const graph = parser.parseModel('no info');
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('Identity');
  });
});
