import { describe, it, expect } from 'vitest';
import { H2OMapper } from '../../../src/mmdnn/h2o/mapper.js';

describe('H2OMapper', () => {
  it('should map h2o', () => {
    const mapper = new H2OMapper({ algo: 'xgboost' });
    const g = mapper.map();
    expect(g.nodes.length).toBe(1);
    expect(g.nodes[0].opType).toBe('TreeEnsembleRegressor');
  });
});
