import { describe, it, expect } from 'vitest';
import { ScikitLearnParser } from '../../../src/mmdnn/scikitlearn/parser.js';

describe('ScikitLearnParser', () => {
  it('should parse RandomForestClassifier', () => {
    const parser = new ScikitLearnParser();
    const graph = parser.parseModel(JSON.stringify({ model: 'RandomForestClassifier' }));
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('TreeEnsembleClassifier');
    expect(graph.nodes[0].attributes['nodes_treeids']).toBeDefined();
    expect(graph.nodes[0].attributes['classlabels_int64s']).toBeDefined();
  });

  it('should parse SVC', () => {
    const parser = new ScikitLearnParser();
    const graph = parser.parseModel(JSON.stringify({ model: 'SVC', kernel: 'rbf', C: 1.0 }));
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('SVMClassifier');
    expect(graph.nodes[0].attributes['kernel_type'].value).toBe('RBF');
    expect(graph.nodes[0].attributes['coefficients']).toBeDefined();
  });

  it('should parse SVC without kernel or C', () => {
    const parser = new ScikitLearnParser();
    const graph = parser.parseModel(JSON.stringify({ model: 'SVC' }));
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('SVMClassifier');
    expect(graph.nodes[0].attributes['kernel_type'].value).toBe('RBF');
  });

  it('should parse LinearClassifier as fallback', () => {
    const parser = new ScikitLearnParser();
    const graph = parser.parseModel(JSON.stringify({ model: 'Unknown' }));
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('LinearClassifier');
    expect(graph.nodes[0].attributes['coefficients']).toBeDefined();
  });

  it('should fallback to Identity on invalid JSON', () => {
    const parser = new ScikitLearnParser();
    const graph = parser.parseModel('invalid json');
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('Identity');
  });
});
