import { Graph } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { optimize, Quantizer, quantize, simplify } from '../src/index.js';

describe('Optimum', () => {
  it('should optimize', async () => {
    const g = new Graph('test');
    g.outputs.push({ name: 'out', shape: [], dtype: 'float32' } as any);
    g.nodes.push({
      opType: 'Identity',
      inputs: ['in'],
      outputs: ['tmp'],
    } as any);
    g.nodes.push({ opType: 'Conv', inputs: ['tmp'], outputs: ['out'] } as any);

    const o = await optimize(g);
    expect(o.nodes.length).toBeGreaterThan(0);
  });

  it('should simplify', async () => {
    const g = new Graph('test');
    const s = await simplify(g);
    expect(s.nodes.length).toBe(0);
  });

  it('should quantize', async () => {
    const g = new Graph('test');
    g.tensors.w = { dtype: 'float32' } as any;
    g.initializers.push('w');

    const q = await quantize(g);
    expect(q.tensors.w.dtype).toBe('int8');

    const quantizer = new Quantizer();
    const q2 = await quantizer.quantize(g, {});
    expect(q2).toBeDefined();
  });
});
