import { describe, it, expect } from 'vitest';
import { ValidationSuite } from '../src/passes/validation.js';

describe('Validation Pass', () => {
  it('should compare ORT vs WVM', async () => {
    const onnxModelBuffer = new ArrayBuffer(0);
    const wvmBytecode = new Uint8Array(0);
    const result = await ValidationSuite.compareORTvsWVM(onnxModelBuffer, wvmBytecode);
    expect(result).toBe(true);
  });
});
