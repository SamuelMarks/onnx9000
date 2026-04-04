import { describe, it, expect } from 'vitest';
import { Graph, ValueInfo, Tensor } from '@onnx9000/core';
import { Quantizer } from '../src/quantization/quantizer';
import { EdgeTPUOptimizer } from '../src/optimizations/edgetpu';

describe('Exporter extra 4', () => {
  it('quantizer missing node', () => {
    const graph = new Graph();
    graph.nodes.push(null as any);

    const q = new Quantizer(graph, { mode: 'int8' });
    q.quantize();
  });

  it('edgetpu float64 check', () => {
    const graph = new Graph();
    graph.valueInfo.push(new ValueInfo('f', [1], 'float64'));
    const opt = new EdgeTPUOptimizer(graph);
    const warnings = opt.optimize();
    expect(warnings.join(' ')).toContain('Float32 tensors');
  });
});
