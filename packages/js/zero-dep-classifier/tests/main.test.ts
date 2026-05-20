import { describe, it, expect } from 'vitest';
import { ZeroDepClassifier } from '../src/index';

describe('ZeroDepClassifier', () => {
  it('should process correctly', () => {
    const obj = new ZeroDepClassifier();
    expect(obj.process('test')).toBe('Zero Dep Classifier processed test');
  });
});
