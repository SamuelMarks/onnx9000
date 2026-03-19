import { describe, it, expect } from 'vitest';
import { compileModel } from '../src/cli.js';

describe('End-to-End Audio Validation', () => {
  it('191-200. should validate Audio compilation', async () => {
    await expect(
      compileModel('whisper.onnx', {
        targetBackend: 'standalone-js',
        dumpMlir: false,
        optimizeLevel: 'O1',
      }),
    ).resolves.not.toThrow();
  });
});
