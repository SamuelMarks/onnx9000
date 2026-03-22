import { describe, it } from 'vitest';
import { TFLiteExporter } from '../src/exporter';
import { compileGraphToTFLite } from '../src/compiler/subgraph';
import { FlatBufferReader } from '../src/flatbuffer/reader';
import { Graph, Tensor, Node } from '@onnx9000/core';
import { BuiltinOperator } from '../src/flatbuffer/schema';

describe('debug', () => {
  it('debug', () => {
    const exporter = new TFLiteExporter();
    const graph = new Graph('TestOpGraph');
    graph.tensors['X'] = new Tensor('X', [1, 10], 'float32', false);
    graph.tensors['Y'] = new Tensor('Y', [1, 10], 'float32', false);
    graph.tensors['Z'] = new Tensor('Z', [10, 1], 'float32', false);
    graph.inputs.push({ name: 'X', shape: [1, 10], dtype: 'float32', id: '0' });
    graph.outputs.push({ name: 'Z', shape: [10, 1], dtype: 'float32', id: '2' });
    graph.nodes.push(new Node('Relu', ['X'], ['Y'], {}, 'relu1'));
    graph.nodes.push(new Node('Reshape', ['Y'], ['Z'], {}, 'resh1'));
    const subgraphsOffset = compileGraphToTFLite(graph, exporter, true);

    console.log('exporter.operatorCodes:', exporter['operatorCodes']);
    console.log('exporter.operatorCodeOffsets:', exporter['operatorCodeOffsets']);
  });
});
