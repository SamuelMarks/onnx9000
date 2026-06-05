import { describe, it, expect } from 'vitest';
import { LibSVMMapper } from '../../../src/mmdnn/libsvm/mapper.js';

describe('LibSVMMapper', () => {
  it('should map libsvm', () => {
    const mapper = new LibSVMMapper({ svmType: 'c_svc', kernelType: 'rbf', rho: 0, coefs: [] });
    const g = mapper.map();
    expect(g.nodes[0].opType).toBe('SVMClassifier');
  });
});
