import { describe, expect, it } from 'vitest';
import { SKL2ONNXConverter } from '../src/index.js';

describe('SKL2ONNXConverter', () => {
  it('should convert scikit-learn model strings and map estimators', () => {
    const conv = new SKL2ONNXConverter();
    expect(conv.convert('rf_clf')).toBe('[ONNX-IR] from skl rf_clf');
    expect(() => conv.convert('')).toThrow('Invalid model string');

    expect(conv.mapEstimator('RandomForestClassifier')).toBe('TreeEnsembleClassifier');
    expect(conv.mapEstimator('CustomEstimator')).toBe('CustomEstimator');
  });
});
