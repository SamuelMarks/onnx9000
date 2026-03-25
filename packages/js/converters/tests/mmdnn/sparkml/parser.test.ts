import { describe, it, expect } from 'vitest';
import { SparkMLParser } from '../../../src/mmdnn/sparkml/parser.js';

describe('SparkMLParser', () => {
  it('should parse LogisticRegression to LinearClassifier', () => {
    const parser = new SparkMLParser();
    const config = { class: 'org.apache.spark.ml.classification.LogisticRegression' };
    const graph = parser.parseModel(JSON.stringify(config));
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('LinearClassifier');
    expect(graph.nodes[0].attributes['coefficients']).toBeDefined();
    expect(graph.nodes[0].attributes['classlabels_int64s']).toBeDefined();
  });

  it('should fallback to Identity when not LogisticRegression', () => {
    const parser = new SparkMLParser();
    const config = { class: 'Unknown' };
    const graph = parser.parseModel(JSON.stringify(config));
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('Identity');
  });

  it('should fallback to Identity on invalid JSON', () => {
    const parser = new SparkMLParser();
    const graph = parser.parseModel('invalid json');
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('Identity');
  });
});
