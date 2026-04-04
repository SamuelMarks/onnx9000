import { describe, it, expect, vi } from 'vitest';
import { Graph, Node, Tensor, Attribute } from '@onnx9000/core';
import { Quantizer } from '../src/quantization/quantizer';

describe('Quantizer Coverage', () => {
  it('should handle customQuantizationMap and fused_activation', () => {
    const graph = new Graph();
    const scaleTensor = new Tensor('scale', [1], 'float32', true, false, new Float32Array([0.5]));
    const zpTensor = new Tensor('zp', [1], 'uint8', true, false, new Uint8Array([128]));
    graph.tensors['scale'] = scaleTensor;
    graph.tensors['zp'] = zpTensor;

    const node1 = new Node('QuantizeLinear', ['x', 'scale', 'zp'], ['y']);
    node1.attributes['fused_activation'] = new Attribute('fused_activation', 'STRING', 'Relu');

    const node2 = new Node('QuantizeLinear', ['x', 'scale', 'zp'], ['y2']);
    node2.attributes['fused_activation'] = new Attribute('fused_activation', 'STRING', 'Relu6');

    const node3 = new Node('DynamicQuantizeLinear', ['x', 'scale', 'zp'], ['y3']);

    const scaleTensor2 = new Tensor(
      'scale2',
      [2],
      'float32',
      true,
      false,
      new Float32Array([0.5, 0.5]),
    );
    const zpTensor2 = new Tensor('zp2', [2], 'uint8', true, false, new Uint8Array([128, 128]));
    graph.tensors['scale2'] = scaleTensor2;
    graph.tensors['zp2'] = zpTensor2;

    // Per-channel warning
    const node5 = new Node('DynamicQuantizeLinear', ['x', 'scale2', 'zp2'], ['y5']);

    graph.nodes.push(node1, node2, node3, node5);

    const quantizer = new Quantizer(graph, {
      mode: 'int8',
      customQuantizationMap: {
        custom_tensor: {
          min: [0],
          max: [1],
          scale: [1],
          zeroPoint: [0],
          quantizedDimension: 0,
        },
      },
    });

    quantizer.quantize();

    expect((quantizer as any).quantizationMap.get('custom_tensor')).toBeDefined();
    expect((quantizer as any).quantizationMap.get('y')).toBeDefined();

    // Check getQuantizationOffset with mock builder
    const builder = {
      startVector: vi.fn(),
      endVector: vi.fn().mockReturnValue(10),
      addFloat32: vi.fn(),
      addInt64: vi.fn(),
      writeInt32: vi.fn(),
      startObject: vi.fn(),
      addFieldOffset: vi.fn(),
      addFieldInt8: vi.fn(),
      addFieldInt32: vi.fn(),
      endObject: vi.fn().mockReturnValue(20),
    };
    const offset = quantizer.getQuantizationOffset(builder as any, new Tensor('y', [1]));
    expect(offset).toBeGreaterThan(0);

    // Test custom tensor offset encoding (has min/max)
    const offset2 = quantizer.getQuantizationOffset(
      builder as any,
      new Tensor('custom_tensor', [1]),
    );
    expect(offset2).toBeGreaterThan(0);

    // Test empty map
    const offset3 = quantizer.getQuantizationOffset(builder as any, new Tensor('none', [1]));
    expect(offset3).toBe(0);
  });
});

it('quantize FP16 and edge cases', () => {
  const graph = new Graph();
  const tFloat = new Tensor(
    'tf',
    [1],
    'float32',
    true,
    false,
    new Float32Array([1.0, 0.0, -1.0, 100000.0, 1e-10]),
  );
  graph.tensors['tf'] = tFloat;

  const quantizerFP16 = new Quantizer(graph, { mode: 'fp16' });
  quantizerFP16.quantize();

  expect(tFloat.dtype).toBe('float16');

  const quantizerNone = new Quantizer(graph, { mode: 'none' });
  quantizerNone.quantize();

  const quantizerInt8Empty = new Quantizer(new Graph(), { mode: 'int8' });
  quantizerInt8Empty.quantize();
});
