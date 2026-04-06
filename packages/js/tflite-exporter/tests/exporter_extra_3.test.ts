import { describe, it, expect } from 'vitest';
import { Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import { Quantizer } from '../src/quantization/quantizer';
import { EdgeTPUOptimizer } from '../src/optimizations/edgetpu';
import { TFLiteExporter } from '../src/exporter';
import { BuiltinOperator } from '../src/flatbuffer/schema';

describe('Exporter extra 3', () => {
  it('exporter json', () => {
    const exp = new TFLiteExporter();
    exp.getOrAddOperatorCode(BuiltinOperator.ADD);
    const j = exp.toJSON();
    expect(j.operatorCodes.length).toBeGreaterThan(0);
  });

  it('quantizer missing node and nan', () => {
    const graph = new Graph();
    graph.nodes.push(null as Object);

    const tf = new Tensor('tf', [1], 'float32', true, false, new Float32Array([NaN, Infinity]));
    graph.tensors['tf'] = tf;

    const q = new Quantizer(graph, { mode: 'fp16' });
    q.quantize();

    expect(tf.dtype).toBe('float16');
  });

  it('edgetpu float check', () => {
    const graph = new Graph();
    graph.valueInfo.push(new ValueInfo('f', [1], 'float32'));
    const opt = new EdgeTPUOptimizer(graph);
    const warnings = opt.optimize();
    expect(warnings.join(' ')).toContain('Float32 tensors');
  });
});
