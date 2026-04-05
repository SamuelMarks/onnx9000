import { describe, it, expect, vi } from 'vitest';
import { createTensorRTSession } from '../src/tensorrt';
import { Graph } from '@onnx9000/core';

describe('TensorRT wrapper coverage', () => {
  it('handles tensorrt missing error', async () => {
    // Dynamic import will fail because @onnx9000/tensorrt is not installed properly or ffi is mocked/missing
    const g = new Graph();
    await expect(createTensorRTSession(g)).rejects.toThrow(
      'TensorRT provider requires Node.js and ffi-napi:',
    );
  });

  it('handles tensorrt success', async () => {
    vi.doMock('@onnx9000/tensorrt', () => ({
      TensorRTProvider: class {
        constructor(g: any) {}
      },
    }));

    // We must re-import because vi.doMock works dynamically for subsequent imports
    const mod = await import('../src/tensorrt');
    const g = new Graph();
    const session = await mod.createTensorRTSession(g);
    expect(session).toBeDefined();

    vi.doUnmock('@onnx9000/tensorrt');
  });
});
