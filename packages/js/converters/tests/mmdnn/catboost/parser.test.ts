import { describe, it, expect } from 'vitest';
import { CatBoostParser } from '../../../src/mmdnn/catboost/parser.js';

describe('CatBoostParser', () => {
  it('should parse valid catboost json to TreeEnsembleClassifier', () => {
    const parser = new CatBoostParser();
    const config = { catboost_version: '1.0' };
    const graph = parser.parseModel(JSON.stringify(config));
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('TreeEnsembleClassifier');
    expect(graph.nodes[0].attributes['nodes_treeids']).toBeDefined();
    expect(graph.nodes[0].attributes['classlabels_int64s']).toBeDefined();
  });

  it('should fallback to Identity when not catboost', () => {
    const parser = new CatBoostParser();
    const config = { other: '1.0' };
    const graph = parser.parseModel(JSON.stringify(config));
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('Identity');
  });

  it('should fallback to Identity on invalid JSON', () => {
    const parser = new CatBoostParser();
    const graph = parser.parseModel('invalid json');
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('Identity');
  });
});
