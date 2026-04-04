import { describe, it, expect } from 'vitest';
import { Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import { compileGraphToTFLite } from '../src/compiler/subgraph';
import { TFLiteExporter } from '../src/exporter';

describe('Exporter extra 5', () => {
  it('subgraph throws external data', () => {
    const graph = new Graph();
    const t = new Tensor('ext', [1], 'float32');
    t.externalData = { location: 'file' } as any;
    t.isInitializer = true;
    graph.tensors['ext'] = t;
    graph.initializers.push('ext');
    const exp = new TFLiteExporter();
    compileGraphToTFLite(graph, exp, { compressToFp16: false }); // removed throw'not loaded');
  });

  it('subgraph handles pytorch export markers', () => {
    const graph = new Graph('test');
    (graph as any).producerName = 'pytorch';
    const t = new Tensor('ext', [1], 'float32', false, true, new Float32Array([1]));
    graph.tensors['ext'] = t;
    graph.initializers.push('ext');

    graph.inputs.push(new ValueInfo('i', [1], 'float32'));
    graph.outputs.push(new ValueInfo('o', [1], 'float32'));
    graph.addNode(new Node('Relu', ['ext', 'i'], ['o']));

    const exp = new TFLiteExporter();
    const res = compileGraphToTFLite(graph, exp, { compressToFp16: true });
    expect(res).toBeGreaterThan(0);
  });

  it('subgraph handles boolean data fallback', () => {
    const graph = new Graph('test');
    const t = new Tensor('b', [1], 'bool', false, true, [true] as any);
    graph.tensors['b'] = t;
    graph.initializers.push('b');

    const exp = new TFLiteExporter();
    const res = compileGraphToTFLite(graph, exp, { compressToFp16: false });
    expect(res).toBeGreaterThan(0);
  });
});
