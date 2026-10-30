import { describe, expect, it } from 'vitest';
import { ZeroDepClassifier } from '../src/index.js';

describe('ZeroDepClassifier', () => {
  it('should normalize RGB pixels and compute top-1 softmax predictions', () => {
    const clf = new ZeroDepClassifier();
    expect(clf.process('cat.jpg')).toBe('Zero Dep Classifier processed cat.jpg');

    const [nr, ng, nb] = clf.normalizePixel(255, 0, 128);
    expect(nr).toBeGreaterThan(2.0);
    expect(ng).toBeLessThan(-2.0);
    expect(typeof nb).toBe('number');

    const pred = clf.predictTop1(new Float32Array([0.0, 0.0]));
    expect(pred.classIndex).toBe(0);
    expect(pred.probability).toBeCloseTo(0.5);

    expect(() => clf.predictTop1(new Float32Array([]))).toThrow('Logits cannot be empty');
  });
});
