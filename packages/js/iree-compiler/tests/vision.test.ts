import { describe, it, expect } from 'vitest';
import { compileModel } from '../src/cli.js';

describe('End-to-End Vision Validation', () => {
  it('171-180. should validate Vision compilation', async () => {
    await expect(
      compileModel('resnet.onnx', { targetBackend: 'wgsl', dumpMlir: true, optimizeLevel: 'O3' }),
    ).resolves.not.toThrow();
  });
});
