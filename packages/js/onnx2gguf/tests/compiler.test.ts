import { describe, it, expect } from 'vitest';
import { compileGGUF } from '../src/compiler.js';
import { Graph } from '@onnx9000/core';

describe('compileGGUF', () => {
  it('should compile', () => {
    const g = new Graph('test');
    const buf = compileGGUF(g);
    expect(buf.byteLength).toBeGreaterThan(0);
  });
});
