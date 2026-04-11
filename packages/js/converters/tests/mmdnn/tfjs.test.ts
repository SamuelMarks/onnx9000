import { describe, test, expect } from 'vitest';
import { Graph, Node, Tensor, ValueInfo, Attribute } from '@onnx9000/core';
import { generateTFJSCode, isLinearGraph } from '../../src/mmdnn/tfjs/generator.js';
import { serializeTFJSWeights } from '../../src/mmdnn/tfjs/serializer.js';

describe('TFJS Generator - isLinearGraph', () => {
  test('linear', () => {
    const graph = new Graph('linear_test');
    graph.inputs.push(new ValueInfo('in', [1, 3, 224, 224], 'float32'));
    graph.outputs.push(new ValueInfo('out2', [1, 64, 112, 112], 'float32'));

    const node1 = new Node('Conv', ['in', 'w1'], ['out1']);
    const node2 = new Node('Relu', ['out1'], ['out2']);

    graph.nodes.push(node1, node2);
    graph.initializers.push('w1');

    expect(isLinearGraph(graph)).toBe(true);
  });

  test('branching', () => {
    const graph = new Graph('branch_test');
    graph.inputs.push(new ValueInfo('in', [1, 3, 224, 224], 'float32'));
    graph.outputs.push(new ValueInfo('out3', [1, 128, 112, 112], 'float32'));

    const node1 = new Node('Conv', ['in', 'w1'], ['out1']);
    const node2 = new Node('Conv', ['in', 'w2'], ['out2']);
    const node3 = new Node('Add', ['out1', 'out2'], ['out3']);

    graph.nodes.push(node1, node2, node3);

    expect(isLinearGraph(graph)).toBe(false);
  });

  test('multiple inputs/outputs', () => {
    const graph1 = new Graph('multi_in');
    graph1.inputs.push(new ValueInfo('in1', [1], 'float32'), new ValueInfo('in2', [1], 'float32'));
    expect(isLinearGraph(graph1)).toBe(false);

    const graph2 = new Graph('multi_out');
    graph2.inputs.push(new ValueInfo('in', [1], 'float32'));
    graph2.outputs.push(
      new ValueInfo('out1', [1], 'float32'),
      new ValueInfo('out2', [1], 'float32'),
    );
    expect(isLinearGraph(graph2)).toBe(false);
  });
});

describe('TFJS Generator - generateTFJSCode', () => {
  test('sequential API for linear model', () => {
    const graph = new Graph('linear_test');
    graph.inputs.push(new ValueInfo('in', [-1, 3, 224, 224], 'float32'));
    graph.outputs.push(new ValueInfo('out2', [-1, 64, 112, 112], 'float32'));

    const node1 = new Node('Conv', ['in', 'w1', 'b1'], ['out1'], {
      strides: new Attribute('strides', 'INTS', [2, 2]),
      pads: new Attribute('pads', 'INTS', [1, 1, 1, 1]),
    });
    const node2 = new Node('Relu', ['out1'], ['out2']);

    graph.nodes.push(node1, node2);
    graph.initializers.push('w1', 'b1');
    graph.tensors['w1'] = new Tensor('w1', [64, 3, 3, 3], 'float32', true);

    const code = generateTFJSCode(graph);
    const graph2 = new Graph('test2');
    graph2.inputs.push({ name: 'in', shape: [-1, 10], type: 'float32' });
    graph2.outputs.push({ name: 'out', shape: [-1, 10], type: 'float32' });
    graph2.nodes.push({
      name: 'n',
      opType: 'UnknownOp',
      inputs: ['in'],
      outputs: ['out'],
      attributes: {},
    });
    expect(() => generateTFJSCode(graph2)).toThrow('Unsupported operator: UnknownOp');
  });

  test('generateLayerCode without tensors (fallback units)', () => {
    const graph = new Graph('fallback_test');
    graph.inputs.push(new ValueInfo('in', [1, 3, 224, 224], 'float32'));
    graph.outputs.push(new ValueInfo('out2', [1, 10], 'float32'));

    // Gemm without weight tensor should fallback to units=1
    const n1 = new Node('Gemm', ['in'], ['out1']);
    // Conv without weight tensor should just not set filters/kernelSize
    const n2 = new Node('Conv', ['out1'], ['out2']);

    graph.nodes.push(n1, n2);

    const code = generateTFJSCode(graph);
    expect(code).toContain("'units':1"); // from fallback
    expect(code).toContain(
      "tf.layers.conv2d({'padding':'valid','dataFormat':'channelsFirst','useBias':false})",
    );
  });
});

describe('TFJS Serializer', () => {
  test('serializeTFJSWeights', () => {
    const graph = new Graph('weight_test');

    const t1 = new Tensor('w1', [2, 2], 'float32', true, true, new Float32Array([1, 2, 3, 4]));
    const t2 = new Tensor('w2', [2], 'float32', true, true, new Float32Array([5, 6]));
    const t3 = new Tensor('w3', [1], 'float32', true, true, new Float32Array([7]));

    graph.tensors['w1'] = t1;
    graph.tensors['w2'] = t2;
    graph.tensors['w3'] = t3;
    graph.initializers.push('w1', 'w2'); // Note: 'w3' is NOT in initializers, but isInitializer is true on the tensor

    const { modelJson, weightsBin } = serializeTFJSWeights(graph);

    expect(weightsBin).toBeInstanceOf(Uint8Array);
    expect(weightsBin.length).toBe(28); // 4 + 2 + 1 floats = 7 floats, 7 * 4 = 28 bytes

    expect((modelJson as Object).weightsManifest[0].weights).toHaveLength(3);
    expect((modelJson as Object).weightsManifest[0].weights[0].name).toBe('w1');
    expect((modelJson as Object).weightsManifest[0].weights[0].shape).toEqual([2, 2]);
    expect((modelJson as Object).weightsManifest[0].weights[1].name).toBe('w2');
    expect((modelJson as Object).weightsManifest[0].weights[2].name).toBe('w3');
  });
});
