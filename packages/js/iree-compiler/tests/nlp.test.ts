import { describe, it, expect } from 'vitest';
import { compileModel } from '../src/cli.js';

describe('End-to-End NLP Validation', () => {
  it('181-190. should validate NLP compilation', async () => {
    await expect(
      compileModel('bert.onnx', { targetBackend: 'wasm', dumpMlir: false, optimizeLevel: 'O3' }),
    ).resolves.not.toThrow();
  });
});
