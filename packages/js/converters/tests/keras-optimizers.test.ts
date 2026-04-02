import { describe, it, expect } from 'vitest';
import {
  KerasGraphOptimizer,
  optimizeFusedOps,
  applyQuantization,
} from '../src/keras/optimizers.js';
import { Graph, Node } from '@onnx9000/core';

describe('KerasGraphOptimizer', () => {
  it('should fuse Conv and BatchNormalization', () => {
    const graph = new Graph('test');
    const conv = new Node('Conv', ['in', 'w', 'b'], ['conv_out'], {}, 'conv1');
    const bn = new Node(
      'BatchNormalization',
      ['conv_out', 'scale', 'bias', 'mean', 'var'],
      ['bn_out'],
      {},
      'bn1',
    );
    graph.nodes.push(conv, bn);

    const optimizer = new KerasGraphOptimizer();
    optimizer.optimize(graph);

    expect(graph.nodes).toHaveLength(1);
    expect(graph.nodes[0].opType).toBe('Conv');
    expect(graph.nodes[0].outputs[0]).toBe('bn_out');
    expect(graph.nodes[0].name).toContain('_fused_bn');
  });

  it('should fuse Gemm and BatchNormalization', () => {
    const graph = new Graph('test');
    const gemm = new Node('Gemm', ['in', 'w', 'b'], ['gemm_out'], {}, 'gemm1');
    const bn = new Node(
      'BatchNormalization',
      ['gemm_out', 'scale', 'bias', 'mean', 'var'],
      ['bn_out'],
      {},
      'bn1',
    );
    graph.nodes.push(gemm, bn);

    const optimizer = new KerasGraphOptimizer();
    optimizer.optimize(graph);

    expect(graph.nodes).toHaveLength(1);
    expect(graph.nodes[0].opType).toBe('Gemm');
    expect(graph.nodes[0].outputs[0]).toBe('bn_out');
  });

  it('should fuse Conv, Add, and Relu', () => {
    const graph = new Graph('test');
    const conv = new Node('Conv', ['in', 'w'], ['conv_out'], {}, 'conv1');
    const add = new Node('Add', ['conv_out', 'bias'], ['add_out'], {}, 'add1');
    const relu = new Node('Relu', ['add_out'], ['relu_out'], {}, 'relu1');
    graph.nodes.push(conv, add, relu);

    const optimizer = new KerasGraphOptimizer();
    optimizer.optimize(graph);

    // fuseConvAddRelu just renames the Relu node if it finds the pattern
    const reluNode = graph.nodes.find((n) => n.opType === 'Relu');
    expect(reluNode?.name).toContain('_fused_conv_add');
  });

  it('should remove Identity nodes', () => {
    const graph = new Graph('test');
    const input = new Node('Input', [], ['in'], {}, 'input');
    const identity = new Node('Identity', ['in'], ['id_out'], {}, 'identity');
    const relu = new Node('Relu', ['id_out'], ['out'], {}, 'relu');
    graph.nodes.push(input, identity, relu);

    const optimizer = new KerasGraphOptimizer();
    optimizer.optimize(graph);

    expect(graph.nodes.find((n) => n.opType === 'Identity')).toBeUndefined();
    expect(relu.inputs[0]).toBe('in');
  });

  it('should eliminate redundant Reshapes', () => {
    const graph = new Graph('test');
    const r1 = new Node('Reshape', ['in', 's1'], ['r1_out'], {}, 'r1');
    const r2 = new Node('Reshape', ['r1_out', 's2'], ['out'], {}, 'r2');
    graph.nodes.push(r1, r2);

    const optimizer = new KerasGraphOptimizer();
    optimizer.optimize(graph);

    expect(graph.nodes).toHaveLength(1);
    expect(graph.nodes[0].inputs[0]).toBe('in');
  });

  it('should handle optimizeFusedOps', () => {
    const nodes = [
      { opType: '_FusedConv2D', name: 'fused', outputs: ['out'], inputs: ['in'], attributes: [] },
      {
        opType: '_FusedMatMul',
        name: 'fused_matmul',
        outputs: ['out2'],
        inputs: ['in2'],
        attributes: [],
      },
      { opType: 'StopGradient', name: 'stop', outputs: ['out3'], inputs: ['in3'], attributes: [] },
    ];
    const optimized = optimizeFusedOps(nodes as any);
    expect(optimized.some((n) => n.opType === 'Conv')).toBe(true);
    expect(optimized.some((n) => n.opType === 'Relu')).toBe(true);
    expect(optimized.some((n) => n.opType === 'MatMul')).toBe(true);
    expect(optimized.find((n) => n.opType === 'StopGradient')).toBeUndefined();
  });

  it('should apply quantization', () => {
    const weights = [{ name: 'w', dtype: 'float32', data: new Float32Array([1, 2, 3]) }];
    const quantized = applyQuantization(weights as any, 'fp16');
    expect(quantized[0].dtype).toBe('fp16');
  });
});
