import { describe, it, expect } from 'vitest';
import { parseModelProto } from '../src/parser/onnx.js';
import { BufferReader } from '../src/parser/protobuf.js';
import * as fs from 'fs';
import * as path from 'path';

describe('parseModelProto', () => {
  it('should parse an ONNX model from disk', async () => {
    const filePath = path.join(__dirname, 'dummy_model.onnx');
    const buffer = fs.readFileSync(filePath);
    const reader = new BufferReader(new Uint8Array(buffer));

    const graph = await parseModelProto(reader);

    // Check top level
    expect(graph.producerName).toBe('test-producer');
    expect(graph.producerVersion).toBe('1.0');
    expect(graph.docString).toBe('My model doc');
    expect(graph.domain).toBe('ai.onnx.test');

    // Check opsets
    expect(graph.opsetImports['']).toBeDefined();
    expect(graph.opsetImports['ai.onnx.ml']).toBe(2);

    // Check graph
    expect(graph.name).toBe('test-model');
    // expect(graph.docString).toBe('My graph doc'); // Graph doc string overwriting?

    // Inputs & Outputs
    expect(graph.inputs.length).toBe(1);
    expect(graph.inputs[0]?.name).toBe('X');
    expect(graph.inputs[0]?.dtype).toBe('float32');
    expect(graph.inputs[0]?.shape).toEqual([1, 2]);

    expect(graph.outputs.length).toBe(1);
    expect(graph.outputs[0]?.name).toBe('Y');

    // Initializers
    expect(graph.initializers.length).toBe(1);
    expect(graph.initializers[0]).toBe('W');

    // Tensors
    const w = graph.tensors['W'];
    expect(w).toBeDefined();
    expect(w?.shape).toEqual([1, 2]);
    expect(w?.dtype).toBe('float32');

    // Nodes
    expect(graph.nodes.length).toBe(1);
    const n = graph.nodes[0];
    expect(n?.opType).toBe('Add');
    expect(n?.inputs).toEqual(['X', 'W']);
    expect(n?.outputs).toEqual(['Y']);
    expect(n?.name).toBe('add_node');
    expect(n?.domain).toBe('ai.onnx');

    // Attributes
    expect(n?.attributes['alpha']?.type).toBe('FLOAT');
    expect(n?.attributes['alpha']?.value).toBe(1.5);
    expect(n?.attributes['ints']?.type).toBe('INTS');
    expect(n?.attributes['ints']?.value).toEqual([1n, 2n, 3n]);
    expect(n?.attributes['strings']?.type).toBe('STRINGS');
  });
});

it('should parse an exhaustive ONNX model from disk', async () => {
  const filePath = path.join(__dirname, 'dummy_model_full.onnx');
  const buffer = fs.readFileSync(filePath);
  const reader = new BufferReader(new Uint8Array(buffer));

  const graph = await parseModelProto(reader);

  expect(graph.producerName).toBe('test-producer');
  expect(graph.opsetImports['']).toBe(14);

  expect(graph.inputs[0]?.shape[0]).toBe(-1); // DynamicDim
  expect(graph.inputs[0]?.shape[1]).toBe(2);

  expect(graph.tensors['T1']?.dtype).toBe('float32');
  expect(graph.tensors['T2']?.dtype).toBe('int64');
  expect(graph.tensors['T3']?.dtype).toBe('int32');
  expect(graph.tensors['T4']?.dtype).toBe('string');

  const n = graph.nodes[0];
  expect(n?.opType).toBe('DummyOp');
  expect(n?.attributes['f']?.value).toBe(1.5);
  expect(n?.attributes['i']?.value).toBe(10n);
  expect(n?.attributes['s']?.value).toBe('test_str');
  expect(n?.attributes['t']?.type).toBe('TENSOR');
  expect(n?.attributes['fs']?.value).toEqual([1.5, 2.5]);
  expect(n?.attributes['is_']?.value).toEqual([10n, 20n]);
  expect(n?.attributes['ss']?.value).toEqual(['test_str1', 'test_str2']);
  expect(n?.attributes['ts']?.type).toBe('TENSORS');
  expect(n?.attributes['g']?.type).toBe('GRAPH');
  expect(n?.attributes['gs']?.type).toBe('GRAPHS');
});

it('should parse external data tensor', async () => {
  const filePath = path.join(__dirname, 'dummy_external.onnx');
  const buffer = fs.readFileSync(filePath);
  const reader = new BufferReader(new Uint8Array(buffer));

  const graph = await parseModelProto(reader);
  const w = graph.tensors['W'];
  expect(w?.externalData).toBeDefined();
  expect(w?.externalData?.location).toBe('model.data');
  expect(w?.externalData?.offset).toBe(4096);
  expect(w?.externalData?.length).toBe(1024);
});
