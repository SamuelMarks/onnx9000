import { describe, it, expect } from 'vitest';
import {
  BufferWriter,
  WIRE_TYPE_VARINT,
  WIRE_TYPE_64BIT,
  WIRE_TYPE_LENGTH_DELIMITED,
  WIRE_TYPE_32BIT,
} from '../src/parser/protobuf_writer.js';
import { serializeModelProto } from '../src/parser/onnx_writer.js';
import { Graph, ValueInfo } from '../src/ir/graph.js';
import { Node, Attribute } from '../src/ir/node.js';
import { Tensor } from '../src/ir/tensor.js';
import {
  BufferReader,
  readTag,
  readVarInt,
  readString,
  readVarInt64,
} from '../src/parser/protobuf.js';
import { parseModelProto } from '../src/parser/onnx.js';

describe('BufferWriter', () => {
  it('should write bytes and ensure space', () => {
    const writer = new BufferWriter(2);
    writer.writeByte(1);
    writer.writeByte(2);
    writer.writeByte(3); // should trigger ensureSpace
    const result = writer.getResult();
    expect(result).toEqual(new Uint8Array([1, 2, 3]));
  });

  it('should write bytes array', () => {
    const writer = new BufferWriter();
    writer.writeBytes(new Uint8Array([10, 20]));
    expect(writer.getResult()).toEqual(new Uint8Array([10, 20]));
  });

  it('should write varints', async () => {
    const writer = new BufferWriter();
    writer.writeVarInt(1);
    writer.writeVarInt(128);
    writer.writeVarInt(-1); // should call writeVarInt64
    const reader = new BufferReader(writer.getResult());
    expect(await readVarInt(reader)).toBe(1);
    expect(await readVarInt(reader)).toBe(128);
    expect(await readVarInt64(reader)).toBe(BigInt('18446744073709551615')); // 64-bit -1
  });

  it('should write varint64', async () => {
    const writer = new BufferWriter();
    writer.writeVarInt64(1n);
    writer.writeVarInt64(128n);
    writer.writeVarInt64(-1n);
    const reader = new BufferReader(writer.getResult());
    expect(await readVarInt64(reader)).toBe(1n);
    expect(await readVarInt64(reader)).toBe(128n);
    expect(await readVarInt64(reader)).toBe(BigInt('18446744073709551615'));
  });

  it('should write strings', async () => {
    const writer = new BufferWriter();
    writer.writeString('hello');
    const reader = new BufferReader(writer.getResult());
    const len = await readVarInt(reader);
    expect(len).toBe(5);
    expect(await readString(reader, 5)).toBe('hello');
  });

  it('should write tags', async () => {
    const writer = new BufferWriter();
    writer.writeTag(1, WIRE_TYPE_VARINT);
    const reader = new BufferReader(writer.getResult());
    const tag = await readTag(reader);
    expect(tag).toEqual({ fieldNumber: 1, wireType: WIRE_TYPE_VARINT });
  });

  it('should write float', () => {
    const writer = new BufferWriter();
    writer.writeFloat(1.5);
    const result = writer.getResult();
    expect(result.length).toBe(4);
    const view = new DataView(result.buffer);
    expect(view.getFloat32(0, true)).toBe(1.5);
  });

  it('should write int64 fixed', () => {
    const writer = new BufferWriter();
    writer.writeInt64(123n);
    const result = writer.getResult();
    expect(result.length).toBe(8);
    const view = new DataView(result.buffer);
    expect(view.getBigInt64(0, true)).toBe(123n);
  });
});

describe('onnx_writer', () => {
  it('should serialize a simple model', async () => {
    const graph = new Graph('test_graph');
    graph.producerName = 'test_producer';

    graph.inputs.push(new ValueInfo('input', [1, 3, 224, 224], 'float32'));
    graph.outputs.push(new ValueInfo('output', [1, 10], 'float32'));

    const node = new Node('Relu', ['input'], ['output'], {}, 'relu1');
    graph.nodes.push(node);

    const bytes = serializeModelProto(graph, 17);
    expect(bytes).toBeInstanceOf(Uint8Array);
    expect(bytes.length).toBeGreaterThan(0);

    // Parse it back
    const reader = new BufferReader(bytes);
    const parsedGraph = await parseModelProto(reader);
    expect(parsedGraph.name).toBe('test_graph');
    expect(parsedGraph.producerName).toBe('test_producer');
    expect(parsedGraph.nodes.length).toBe(1);
    expect(parsedGraph.nodes[0].opType).toBe('Relu');
    expect(parsedGraph.inputs[0].name).toBe('input');
    expect(parsedGraph.outputs[0].name).toBe('output');
  });

  it('should handle initializers and attributes', async () => {
    const graph = new Graph('attr_test');

    const tensor = new Tensor(
      'weight',
      [2, 2],
      'float32',
      true,
      false,
      new Float32Array([1, 2, 3, 4]),
    );
    graph.tensors['weight'] = tensor;
    graph.initializers.push('weight');

    const node = new Node(
      'Conv',
      ['input', 'weight'],
      ['output'],
      {
        kernel_shape: new Attribute('kernel_shape', 'INTS', [3, 3]),
        epsilon: new Attribute('epsilon', 'FLOAT', 1e-5),
        int_val: new Attribute('int_val', 'INT', 42),
        str_val: new Attribute('str_val', 'STRING', 'test'),
        floats_val: new Attribute('floats_val', 'FLOATS', [1.0, 2.0]),
      },
      'conv1',
    );
    graph.nodes.push(node);

    const bytes = serializeModelProto(graph);
    const reader = new BufferReader(bytes);
    const parsedGraph = await parseModelProto(reader);

    expect(parsedGraph.initializers).toContain('weight');
    const parsedTensor = parsedGraph.tensors['weight'];
    expect(parsedTensor.dtype).toBe('float32');
    expect(parsedTensor.shape).toEqual([2, 2]);
    const actualData = new Float32Array((parsedTensor.data as Uint8Array).slice().buffer);
    expect(Array.from(actualData)).toEqual([1, 2, 3, 4]);

    const parsedNode = parsedGraph.nodes[0];
    expect(parsedNode.attributes['kernel_shape'].value).toEqual([3n, 3n]);
    expect(parsedNode.attributes['epsilon'].value).toBeCloseTo(1e-5);
    expect(parsedNode.attributes['int_val'].value).toBe(42n); // parseModelProto returns bigint for INT
    expect(parsedNode.attributes['str_val'].value).toBe('test');
    expect(parsedNode.attributes['floats_val'].value).toEqual([1.0, 2.0]);
  });

  it('should handle dynamic shapes in ValueInfo', async () => {
    const graph = new Graph('dynamic_test');
    graph.inputs.push(new ValueInfo('input', [-1, 'batch', 224], 'int32'));

    const bytes = serializeModelProto(graph);
    const reader = new BufferReader(bytes);
    const parsedGraph = await parseModelProto(reader);

    expect(parsedGraph.inputs[0].shape).toEqual([-1, 'batch', 224]);
    expect(parsedGraph.inputs[0].dtype).toBe('int32');
  });

  it('should handle all tensor data types', async () => {
    const types: string[] = [
      'float32',
      'uint8',
      'int8',
      'uint16',
      'int16',
      'int32',
      'int64',
      'string',
      'bool',
      'float16',
      'float64',
      'uint32',
      'uint64',
      'bfloat16',
    ];

    for (const type of types) {
      const graph = new Graph(`test_${type}`);
      const tensor = new Tensor(`t_${type}`, [1], type as any, true, false, new Uint8Array([0]));
      graph.tensors[`t_${type}`] = tensor;
      graph.initializers.push(`t_${type}`);

      const bytes = serializeModelProto(graph);
      const reader = new BufferReader(bytes);
      const parsedGraph = await parseModelProto(reader);
      expect(parsedGraph.tensors[`t_${type}`].dtype).toBe(type);
    }
  });

  it('should use default producer name if not provided', async () => {
    const graph = new Graph('default_producer');
    const bytes = serializeModelProto(graph);
    const reader = new BufferReader(bytes);
    const parsedGraph = await parseModelProto(reader);
    expect(parsedGraph.producerName).toBe('onnx9000');
  });

  it('should handle unsupported dtype by defaulting to float32', async () => {
    const graph = new Graph('unsupported_dtype');
    // @ts-ignore
    const tensor = new Tensor(
      'x',
      [1],
      'invalid_dtype' as any,
      true,
      false,
      new Float32Array([1.0]),
    );
    graph.tensors['x'] = tensor;
    graph.initializers.push('x');

    const bytes = serializeModelProto(graph);
    const reader = new BufferReader(bytes);
    const parsedGraph = await parseModelProto(reader);
    expect(parsedGraph.tensors['x'].dtype).toBe('float32');
  });

  it('should handle empty dim name and empty graph name', async () => {
    const g = new Graph('');
    const vi = new ValueInfo('in', [''], 'float32');
    g.inputs.push(vi);
    const bytes = serializeModelProto(g);
    const reader = new BufferReader(bytes);
    const parsedGraph = await parseModelProto(reader);
    expect(parsedGraph.name).toBe('model_graph');
    expect(parsedGraph.inputs[0].shape[0]).toBe('?');
  });
});
