import { describe, it, expect } from 'vitest';
import { TFLiteExporter } from '../src/exporter';
import { compileGraphToTFLite } from '../src/compiler/subgraph';
import { FlatBufferReader } from '../src/flatbuffer/reader';
import { Graph, Tensor } from '@onnx9000/core';
import { BuiltinOperator } from '../src/flatbuffer/schema';

describe('TFLite Compiler - SubGraph Mapping', () => {
  it('should compile an empty graph to a valid TFLite SubGraph', () => {
    const exporter = new TFLiteExporter();
    const graph = new Graph('TestGraph');

    // Add some tensors
    const inputTensor = new Tensor('input1', [1, 3, 224, 224], 'float32', false);
    const outputTensor = new Tensor('output1', [1, 1000], 'float32', false);
    const weightTensor = new Tensor(
      'weight1',
      [1000, 3, 3, 3],
      'float32',
      true,
      false,
      new Float32Array([1.0, 2.0, 3.0, 4.0]),
    );

    graph.tensors['input1'] = inputTensor;
    graph.tensors['output1'] = outputTensor;
    graph.tensors['weight1'] = weightTensor;

    graph.inputs.push({ name: 'input1', shape: [1, 3, 224, 224], dtype: 'float32', id: '0' });
    graph.outputs.push({ name: 'output1', shape: [1, 1000], dtype: 'float32', id: '1' });

    const subgraphsOffset = compileGraphToTFLite(graph, exporter);

    // finish by manually putting subgraphs in a vector
    exporter.builder.startVector(4, 1, 4);
    exporter.builder.addOffset(subgraphsOffset);
    const subgraphsVecOffset = exporter.builder.endVector(1);

    const buf = exporter.finish(subgraphsVecOffset, 'test_graph_compilation');

    const reader = new FlatBufferReader(buf);
    expect(reader.checkMagicBytes('TFL3')).toBe(true);

    const modelOffset = reader.getRoot();
    const subgraphsVec = reader.getIndirectOffset(modelOffset, 2); // subgraphs is field 2

    expect(subgraphsVec).not.toBe(0);
    const numSubgraphs = reader.getVectorLength(subgraphsVec);
    expect(numSubgraphs).toBe(1);

    const subgraphOffset = reader.getVectorItemOffset(subgraphsVec, 0); // vector items are offsets
    const subgraphLoc = subgraphOffset + reader.view.getUint32(subgraphOffset, true); // dereference to subgraph object

    // Subgraph name is field 4
    const name = reader.getString(subgraphLoc, 4);
    expect(name).toBe('TestGraph');

    // Subgraph inputs is field 1
    const inputsVec = reader.getIndirectOffset(subgraphLoc, 1);
    expect(reader.getVectorLength(inputsVec)).toBe(1);

    // Subgraph outputs is field 2
    const outputsVec = reader.getIndirectOffset(subgraphLoc, 2);
    expect(reader.getVectorLength(outputsVec)).toBe(1);

    // Subgraph tensors is field 0
    const tensorsVec = reader.getIndirectOffset(subgraphLoc, 0);
    expect(reader.getVectorLength(tensorsVec)).toBe(3); // input1, output1, weight1
  });

  it('should compile a graph with an elementwise operator', () => {
    const exporter = new TFLiteExporter();
    const graph = new Graph('TestGraph');

    // Add some tensors
    graph.tensors['A'] = new Tensor('A', [1, 10], 'float32', false);
    graph.tensors['B'] = new Tensor('B', [1, 10], 'float32', false);
    graph.tensors['C'] = new Tensor('C', [1, 10], 'float32', false);

    graph.inputs.push({ name: 'A', shape: [1, 10], dtype: 'float32', id: '0' });
    graph.inputs.push({ name: 'B', shape: [1, 10], dtype: 'float32', id: '1' });
    graph.outputs.push({ name: 'C', shape: [1, 10], dtype: 'float32', id: '2' });

    // Add Node
    graph.nodes.push({
      id: 'node1',
      opType: 'Add',
      name: 'add1',
      domain: '',
      inputs: ['A', 'B'],
      outputs: ['C'],
      attributes: {},
      docString: '',
    });

    const subgraphsOffset = compileGraphToTFLite(graph, exporter);

    exporter.builder.startVector(4, 1, 4);
    exporter.builder.addOffset(subgraphsOffset);
    const subgraphsVecOffset = exporter.builder.endVector(1);

    const buf = exporter.finish(subgraphsVecOffset, 'test_add');
    const reader = new FlatBufferReader(buf);

    const modelOffset = reader.getRoot();

    // Check opcodes
    const opCodesVec = reader.getIndirectOffset(modelOffset, 1);
    expect(reader.getVectorLength(opCodesVec)).toBe(1);
    const opCodeOffset = reader.getVectorItemOffset(opCodesVec, 0);
    const opCodeObj = opCodeOffset + reader.view.getUint32(opCodeOffset, true);
    const builtinCode = reader.getInt8(opCodeObj, 0);
    expect(builtinCode).toBe(BuiltinOperator.ADD);

    // Check operators
    const subgraphsVec = reader.getIndirectOffset(modelOffset, 2);
    const subgraphOffset = reader.getVectorItemOffset(subgraphsVec, 0);
    const subgraphLoc = subgraphOffset + reader.view.getUint32(subgraphOffset, true);

    const operatorsVec = reader.getIndirectOffset(subgraphLoc, 3); // field 3
    expect(reader.getVectorLength(operatorsVec)).toBe(1);

    const opOffset = reader.getVectorItemOffset(operatorsVec, 0);
    const opObj = opOffset + reader.view.getUint32(opOffset, true);

    const opIdx = reader.getInt32(opObj, 0);
    expect(opIdx).toBe(0); // Uses first opcode

    const inputsVec = reader.getIndirectOffset(opObj, 1);
    expect(reader.getVectorLength(inputsVec)).toBe(2);

    const outputsVec = reader.getIndirectOffset(opObj, 2);
    expect(reader.getVectorLength(outputsVec)).toBe(1);
  });
});
