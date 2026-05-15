import { describe, it, expect } from 'vitest';
import { loadProgressive, ProgressiveSession } from '../src/progressive.js';
import { Tensor } from '@onnx9000/core';

describe('Progressive Model Loading', () => {
  it('should initialize a ProgressiveSession via loadProgressive', async () => {
    const session = await loadProgressive('https://models.onnx9000.ai/llama3-8b.onnx', {
      maxChunkSize: 1024 * 1024,
    });
    expect(session).toBeInstanceOf(ProgressiveSession);
    expect(session.isLoaded).toBe(false);
  });

  it('should lazy load weights on first run', async () => {
    const session = await loadProgressive('https://models.onnx9000.ai/llama3-8b.onnx');
    const inputs: Record<string, Tensor> = {};
    const outputs = await session.run(inputs);

    expect(outputs).toEqual({});
    expect(session.isLoaded).toBe(true);
  });

  it('should throw error for invalid URLs that are not absolute or http', async () => {
    const session = await loadProgressive('relative_file.onnx');
    const inputs: Record<string, Tensor> = {};
    await expect(session.run(inputs)).rejects.toThrow(/valid HTTP URL or absolute path/);
  });

  it('should handle chunk size options seamlessly on subsequent runs', async () => {
    const session = await loadProgressive('http://localhost/model.onnx', { maxChunkSize: 512 });
    const inputs: Record<string, Tensor> = {};

    await session.run(inputs);
    expect(session.isLoaded).toBe(true);

    // running again should not throw and use isLoaded cache
    const outputs = await session.run(inputs);
    expect(outputs).toEqual({});
  });
});
