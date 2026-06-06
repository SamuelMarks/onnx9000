import { Graph } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { compileGGUF } from '../src/compiler.js';

describe('compileGGUF', () => {
  it('should compile', () => {
    const g = new Graph('test');
    const buf = compileGGUF(g);
    expect(buf.byteLength).toBeGreaterThan(0);
  });
});
