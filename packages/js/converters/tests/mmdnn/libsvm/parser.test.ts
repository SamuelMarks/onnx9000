import { describe, it, expect } from 'vitest';
import { parseLibSVM } from '../../../src/mmdnn/libsvm/parser.js';

describe('libsvm parser', () => {
  it('should parse libsvm', () => {
    const res = parseLibSVM('svm_type c_svc\nkernel_type rbf\nrho 0.5\nSV\n1 1:1');
    expect(res.svmType).toBe('c_svc');
    expect(res.kernelType).toBe('rbf');
    expect(res.rho).toBe(0.5);
    expect(res.coefs[0]).toBe(1);
  });
});
