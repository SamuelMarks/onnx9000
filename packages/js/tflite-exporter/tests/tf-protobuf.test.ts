import { describe, it, expect } from 'vitest';
import { Graph, Node, Tensor } from '@onnx9000/core';
import { TFProtobufEncoder } from '../src/tf-protobuf/encoder';
import { SavedModelGenerator } from '../src/tf-protobuf/generator';

describe('TFLite Compiler - TF Protobuf', () => {
  it('should generate a valid tf representation of the graph', () => {
    const graph = new Graph('TestGraph');
    graph.tensors['X'] = new Tensor('X', [1, 10], 'float32', false);
    graph.tensors['W'] = new Tensor('W', [10, 10], 'float32', true, false, new Float32Array(100));
    graph.tensors['W_int32'] = new Tensor('W_int32', [1], 'int32', true, false, new Int32Array(1));
    graph.tensors['W_int64'] = new Tensor(
      'W_int64',
      [1],
      'int64',
      true,
      false,
      new BigInt64Array(1),
    );
    graph.tensors['W_string'] = new Tensor('W_string', [1], 'string', true, false, ['test']);

    graph.nodes.push(new Node('Add', ['X', 'W'], ['Y'], {}, 'add1'));
    graph.nodes.push(new Node('Mul', ['Y', 'W'], ['Z'], {}, 'mul1'));
    graph.nodes.push(new Node('Relu', ['Z'], ['Out'], {}, 'relu1'));
    graph.nodes.push(new Node('Unknown', ['X'], ['UnknownOut'], {}, 'custom'));

    const generator = new SavedModelGenerator();
    const savedModel = generator.generateFromONNX(graph);

    expect(savedModel.savedModelSchemaVersion).toBe(1);
    expect(savedModel.metaGraphs.length).toBe(1);

    const metaGraph = savedModel.metaGraphs[0]!;
    expect(metaGraph.metaInfoDef.tags).toContain('serve');

    const nodes = metaGraph.graphDef.node;
    expect(nodes.length).toBe(8); // 1 const (W) + 4 ops

    const constNode = nodes.find((n) => n.op === 'Const');
    expect(constNode).toBeDefined();
    expect(constNode!.name).toBe('W');

    const addNode = nodes.find((n) => n.name === 'add1');
    expect(addNode!.op).toBe('AddV2');

    const customNode = nodes.find((n) => n.name === 'custom');
    expect(customNode!.op).toBe('Custom_Unknown');

    const encoder = new TFProtobufEncoder();
    const buf = encoder.encode(savedModel);

    expect(buf.length).toBeGreaterThan(0);
    // test the mock buffer content
    expect(buf[0]).toBe(0x0a);
  });
});
