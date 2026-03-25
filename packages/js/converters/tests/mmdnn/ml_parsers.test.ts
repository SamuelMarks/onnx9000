import { describe, it, expect } from 'vitest';
import { convert } from '../../src/mmdnn/api.js';

describe('MMDNN - ML Framework Parsers Integration', () => {
  it('should parse scikitlearn model', async () => {
    const file = new File([JSON.stringify({ model: 'RandomForestClassifier' })], 'model.json', {
      type: 'text/plain',
    });
    const graph = await convert('scikitlearn', 'onnx', [file]);
    expect(graph.name).toBe('scikitlearn-imported');
    expect(graph.nodes[0].opType).toBe('TreeEnsembleClassifier');
  });

  it('should fallback on malformed scikitlearn model', async () => {
    const file = new File(['invalid json'], 'model.json', { type: 'text/plain' });
    const graph = await convert('scikitlearn', 'onnx', [file]);
    expect(graph.nodes[0].opType).toBe('Identity');
  });

  it('should parse lightgbm model', async () => {
    const file = new File(['tree\nversion=v3\n'], 'model.txt', { type: 'text/plain' });
    const graph = await convert('lightgbm', 'onnx', [file]);
    expect(graph.nodes[0].opType).toBe('TreeEnsembleRegressor');
  });

  it('should fallback on malformed lightgbm model', async () => {
    const file = new File(['other'], 'model.txt', { type: 'text/plain' });
    const graph = await convert('lightgbm', 'onnx', [file]);
    expect(graph.nodes[0].opType).toBe('Identity');
  });

  it('should parse xgboost model', async () => {
    const content = JSON.stringify({ learner: { objective: { name: 'binary:logistic' } } });
    const file = new File([content], 'model.json', { type: 'text/plain' });
    const graph = await convert('xgboost', 'onnx', [file]);
    expect(graph.nodes[0].opType).toBe('TreeEnsembleClassifier');
  });

  it('should parse catboost model', async () => {
    const content = JSON.stringify({ catboost_version: '1.0' });
    const file = new File([content], 'model.json', { type: 'text/plain' });
    const graph = await convert('catboost', 'onnx', [file]);
    expect(graph.nodes[0].opType).toBe('TreeEnsembleClassifier');
  });

  it('should parse sparkml model', async () => {
    const content = JSON.stringify({ class: 'LogisticRegression' });
    const file = new File([content], 'model.json', { type: 'text/plain' });
    const graph = await convert('sparkml', 'onnx', [file]);
    expect(graph.nodes[0].opType).toBe('LinearClassifier');
  });
});
