import { describe, it, expect } from 'vitest';
import { H2OMapper } from '../../../src/mmdnn/h2o/mapper.js';

describe('H2O Mapper', () => {
  it('should map xgboost algo to TreeEnsembleRegressor', () => {
    const mapper = new H2OMapper({ algo: 'xgboost' });
    const graph = mapper.map();
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('TreeEnsembleRegressor');
    expect(graph.nodes[0].attributes['n_targets'].value).toBe(1);
  });

  it('should map deeplearning algo to MatMul', () => {
    const mapper = new H2OMapper({ algo: 'deeplearning' });
    const graph = mapper.map();
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('MatMul');
  });

  it('should map unknown algo to TreeEnsembleRegressor with NONE post_transform', () => {
    const mapper = new H2OMapper({ algo: 'unknown' });
    const graph = mapper.map();
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('TreeEnsembleRegressor');
    expect(graph.nodes[0].attributes['post_transform'].value).toBe('NONE');
  });
});
