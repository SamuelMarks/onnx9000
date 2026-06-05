import { describe, it, expect } from 'vitest';
import { Graph, Node } from '@onnx9000/core';
import { exportModel, optimize, simplify, quantize, Quantizer } from '../src/index.js';

describe('@onnx9000/optimum', () => {
  const createMockGraph = () => {
    const graph = new Graph('TestGraph');
    graph.inputs = [{ name: 'input_1', type: 'tensor', shape: [1, 3, 224, 224] }] as any;
    graph.outputs = [
      { name: 'output_1', type: 'tensor', shape: [1, 1000] },
      { name: 'identity_output', type: 'tensor', shape: [1, 1000] },
    ] as any;
    graph.initializers = ['weight_1', 'other_weight'];
    graph.tensors = {
      weight_1: { dtype: 'float32', shape: [32, 3, 3, 3], buffer: new Float32Array(0) } as any,
      other_weight: { dtype: 'int32', shape: [1], buffer: new Int32Array(0) } as any,
    };

    const convNode = new Node('Conv', ['input_1', 'weight_1'], ['conv_out'], {
      strides: [1, 1],
    } as any);
    const idNode = new Node('Identity', ['conv_out'], ['id_out']);
    const reluNode = new Node('Relu', ['id_out'], ['relu_out']);

    // Dead node that isn't connected to graph outputs
    const deadNode = new Node('Add', ['relu_out', 'other_weight'], ['dead_out']);

    // Dropout node
    const dropoutNode = new Node('Dropout', ['relu_out'], ['drop_out', 'drop_mask']);

    const extraNode = new Node('Extra', ['drop_out'], ['output_1']);

    // Identity connected to graph output
    const outputIdentity = new Node('Identity', ['output_1'], ['identity_output']);

    // Second consumer to prevent Conv+Relu fusion
    const secondRelu = new Node('Relu', ['conv_out'], ['second_relu_out']);
    // Dead node for secondRelu
    const secondDead = new Node('Sigmoid', ['second_relu_out'], ['dead2']);

    graph.nodes.push(
      convNode,
      idNode,
      reluNode,
      deadNode,
      dropoutNode,
      extraNode,
      outputIdentity,
      secondRelu,
      secondDead,
    );
    return graph;
  };

  it('should export model seamlessly', async () => {
    await expect(exportModel('test-model', '/tmp')).resolves.not.toThrow();
  });

  it('should optimize graph by eliminating Identity, Dropout, and DCE', async () => {
    const graph = createMockGraph();
    // Re-create graph to NOT have secondRelu so we CAN test fusion
    graph.nodes = graph.nodes.filter(
      (n) =>
        n.name !== 'secondRelu' &&
        n.name !== 'secondDead' &&
        n.opType !== 'Sigmoid' &&
        n.outputs[0] !== 'second_relu_out',
    );

    const optGraph = await optimize(graph, { disableFusion: false });

    // Nodes expected after DCE, Identity/Dropout removal, and Fusion:
    // 1. ConvRelu (fuses Conv and Relu because 'id_out' identity is removed)
    // 2. Extra (reads directly from ConvRelu output since Dropout is removed)
    // 3. Identity (kept because it's connected to graph output 'identity_output')
    expect(optGraph.nodes.length).toBe(3);

    const fused = optGraph.nodes.find((n) => n.opType === 'ConvRelu');
    expect(fused).toBeDefined();
    expect(fused!.outputs).toContain('relu_out');

    const extra = optGraph.nodes.find((n) => n.opType === 'Extra');
    expect(extra).toBeDefined();
    expect(extra!.inputs).toContain('relu_out'); // Dropout bypassed

    const outputId = optGraph.nodes.find((n) => n.outputs.includes('identity_output'));
    expect(outputId).toBeDefined();
    expect(outputId!.opType).toBe('Identity');
  });

  it('should not fuse Conv+Relu if Conv output has multiple consumers', async () => {
    const graph = createMockGraph();
    // Let's modify Extra to output to 'output_1' AND 'second_relu_out' to force it to be kept
    graph.outputs.push({ name: 'second_relu_out', type: 'tensor', shape: [] } as any);

    const optGraph = await optimize(graph, { disableFusion: false });

    // Because second_relu_out is a graph output, secondRelu is kept.
    // Thus Conv output 'conv_out' is used by Relu AND secondRelu.
    // Fusion should be disabled.
    const conv = optGraph.nodes.find((n) => n.opType === 'Conv');
    expect(conv).toBeDefined();
  });

  it('should simplify graph similarly without fusions', async () => {
    const graph = createMockGraph();
    const simpGraph = await simplify(graph);

    // Identity & Dropout removed, DCE applied. disableFusion is true
    // Surviving nodes: Conv, Relu, Extra, Identity (output_1 -> identity_output)
    expect(simpGraph.nodes.length).toBe(4);
    expect(simpGraph.nodes.find((n) => n.opType === 'Conv')).toBeDefined();
    expect(simpGraph.nodes.find((n) => n.opType === 'Relu')).toBeDefined();
    expect(simpGraph.nodes.find((n) => n.opType === 'Extra')).toBeDefined();
    expect(
      simpGraph.nodes.find((n) => n.opType === 'Identity' && n.outputs.includes('identity_output')),
    ).toBeDefined();
  });

  it('should quantize by modifying weights precision', async () => {
    const graph = createMockGraph();
    const qGraph = await quantize(graph);

    expect(qGraph.tensors['weight_1'].dtype).toBe('int8');

    const quantizer = new Quantizer();
    const qGraph2 = await quantizer.quantize(graph, { method: 'dynamic' });
    expect(qGraph2.tensors['weight_1'].dtype).toBe('int8');
  });
});
