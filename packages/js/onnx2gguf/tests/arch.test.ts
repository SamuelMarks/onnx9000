import { describe, it, expect } from 'vitest';
import { inferArchitecture, extractMetadata } from '../src/arch.js';
import { Graph } from '@onnx9000/core';

describe('arch', () => {
  it('should infer', () => {
    const g = new Graph('mistral');
    expect(inferArchitecture(g)).toBe('mistral');
  });

  it('should extract metadata', () => {
    const g = new Graph('llama');
    const res = extractMetadata(g);
    expect(res).toBeDefined();
  });
});
