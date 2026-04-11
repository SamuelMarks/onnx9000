import { describe, expect, it, vi } from 'vitest';
import { Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import { ONNXNormalizer } from '../../src/mmdnn/verification/normalizer.js';

describe('ONNXNormalizer', () => {
  it('should sanitize input and output names', () => {
    const graph = new Graph('test');
    graph.inputs.push(new ValueInfo('0input', [1, 3, 224, 224], 'float32'));
    graph.outputs.push(new ValueInfo('out.put!', [1, 1000], 'float32'));

    const node = new Node('Relu', ['0input'], ['out.put!'], {}, 'my-node', '');
    graph.addNode(node);

    const normalizer = new ONNXNormalizer();
    normalizer.normalize(graph);

    expect(graph.inputs[0].name).toBe('_0input');
    expect(graph.outputs[0].name).toBe('out_put_');
    expect(graph.nodes[0].inputs[0]).toBe('_0input');
    expect(graph.nodes[0].outputs[0]).toBe('out_put_');
    expect(graph.nodes[0].name).toBe('my_node');
  });

  it('should decompose proprietary ops into ONNX ops', () => {
    const graph = new Graph('test');
    graph.inputs.push(new ValueInfo('input1', [1, 3, 224, 224], 'float32'));
    graph.outputs.push(new ValueInfo('output2', [1, 3, 224, 224], 'float32'));

    const caffeNode = new Node('CaffeScale', ['input1'], ['output1']);
    const mxNetNode = new Node('MxNetActivation', ['output1'], ['output2']);

    graph.addNode(caffeNode);
    graph.addNode(mxNetNode);

    const normalizer = new ONNXNormalizer();
    normalizer.normalize(graph);

    expect(graph.nodes[0].opType).toBe('Mul');
    expect(graph.nodes[1].opType).toBe('Relu');
  });

  it('should convert float64 to float32 globally', () => {
    const graph = new Graph('test');
    const tensor = new Tensor(
      'weight1',
      [1, 1],
      'float64',
      true,
      false,
      new Float64Array([1.5, 2.5]),
    );
    graph.addTensor(tensor);

    const inputInfo = new ValueInfo('input1', [1, 1], 'float64');
    graph.inputs.push(inputInfo);
    graph.outputs.push(new ValueInfo('output1', [1, 1], 'float64'));

    const node = new Node('Add', ['input1', 'weight1'], ['output1']);
    graph.addNode(node);

    const normalizer = new ONNXNormalizer();
    normalizer.normalize(graph);

    expect(graph.tensors['weight1'].dtype).toBe('float32');
    expect(graph.tensors['weight1'].data instanceof Float32Array).toBe(true);
    expect(graph.inputs[0].dtype).toBe('float32');
  });

  it('should detect and remove unconnected subgraphs (islands)', () => {
    const graph = new Graph('test');
    graph.inputs.push(new ValueInfo('input1', [1], 'float32'));
    graph.outputs.push(new ValueInfo('output1', [1], 'float32'));

    // Node 1: Connected to output (Useful)
    const node1 = new Node('Relu', ['input1'], ['output1']);
    // Node 2: Unconnected to output (Island)
    const node2 = new Node('Relu', ['input1'], ['island_out']);
    // Node 3: Unconnected dependency (Island)
    const node3 = new Node('Relu', ['island_out'], ['island_out2']);

    graph.addNode(node1);
    graph.addNode(node2);
    graph.addNode(node3);

    const islandTensor = new Tensor(
      'island_out',
      [1],
      'float32',
      false,
      false,
      new Float32Array([1]),
    );
    graph.addTensor(islandTensor);
    graph.initializers.push('island_out');

    const normalizer = new ONNXNormalizer();
    normalizer.normalize(graph);

    // Only node1 should remain
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].outputs[0]).toBe('output1');

    // Island tensors should be cleaned up
    expect(graph.tensors['island_out']).toBeUndefined();
    expect(graph.initializers).not.toContain('island_out');
  });

  it('should verify parity', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const normalizer = new ONNXNormalizer();
    expect(normalizer.verifyParity()).toBe(true);
    expect(spy).toHaveBeenCalledWith(
      'Parity verification via WebGPU requires full runtime and is skipped by the normalizer.',
    );
    spy.mockRestore();
  });
});
