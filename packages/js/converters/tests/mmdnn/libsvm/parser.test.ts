import { describe, it, expect } from 'vitest';
import { parseLibSVM } from '../../../src/mmdnn/libsvm/parser.js';

describe('LibSVM Parser', () => {
  it('should parse basic text format', () => {
    const text = `
svm_type c_svc
kernel_type rbf
gamma 0.5
nr_class 2
total_sv 2
rho -0.1
label 1 0
nr_sv 1 1
SV
1.0 1:0.5 2:0.1
-1.0 1:0.2 2:0.8
    `;
    const parsed = parseLibSVM(text);
    expect(parsed.svmType).toBe('c_svc');
    expect(parsed.kernelType).toBe('rbf');
    expect(parsed.rho).toBe(-0.1);
    expect(parsed.coefs.length).toBe(2);
    expect(parsed.coefs[0]).toBe(1.0);
    expect(parsed.coefs[1]).toBe(-1.0);
  });
});
