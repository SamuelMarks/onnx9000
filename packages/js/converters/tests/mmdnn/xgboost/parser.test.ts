import { describe, it, expect } from 'vitest';
import { XGBoostParser } from '../../../src/mmdnn/xgboost/parser.js';

describe('XGBoostParser', () => {
  it('should parse logistic learner to TreeEnsembleClassifier', () => {
    const parser = new XGBoostParser();
    const config = { learner: { objective: { name: 'binary:logistic' } } };
    const graph = parser.parseModel(JSON.stringify(config));
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('TreeEnsembleClassifier');
    expect(graph.nodes[0].attributes['nodes_treeids']).toBeDefined();
    expect(graph.nodes[0].attributes['classlabels_int64s']).toBeDefined();
  });

  it('should parse unknown learner to TreeEnsembleRegressor', () => {
    const parser = new XGBoostParser();
    const config = { learner: { objective: { name: 'reg:squarederror' } } };
    const graph = parser.parseModel(JSON.stringify(config));
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('TreeEnsembleRegressor');
    expect(graph.nodes[0].attributes['target_ids']).toBeDefined();
    expect(graph.nodes[0].attributes['n_targets']).toBeDefined();
  });

  it('should fallback to Identity on invalid JSON', () => {
    const parser = new XGBoostParser();
    const graph = parser.parseModel('invalid json');
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('Identity');
  });
});
