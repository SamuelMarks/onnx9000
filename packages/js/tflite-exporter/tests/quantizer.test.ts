import { describe, it, expect } from 'vitest';
import { Graph, Node, Tensor } from '@onnx9000/core';
import { Quantizer } from '../src/quantization/quantizer';

describe('TFLite Compiler - Quantizer', () => {
  it('should downcast float32 to float16', () => {
    const graph = new Graph('TestGraph');
    const data = new Float32Array([1.0, -1.0, 0.0, 2.0, 1000000.0, 0.000001, -1000000.0]);
    graph.tensors['W'] = new Tensor('W', [7], 'float32', true, false, data);

    const quantizer = new Quantizer(graph, { mode: 'fp16' });
    quantizer.quantize();

    expect(graph.tensors['W'].dtype).toBe('float16');
    expect(graph.tensors['W'].data).toBeInstanceOf(Uint16Array);

    const f16 = graph.tensors['W'].data as Uint16Array;
    expect(f16[0]).toBe(0x3c00); // 1.0
    expect(f16[1]).toBe(0xbc00); // -1.0
    expect(f16[2]).toBe(0x0000); // 0.0
    expect(f16[3]).toBe(0x4000); // 2.0
  });

  it('should parse QDQ scales/zeros into QuantizationParameters', () => {
    const graph = new Graph('TestGraph');
    graph.tensors['scale'] = new Tensor(
      'scale',
      [1],
      'float32',
      true,
      false,
      new Float32Array([0.5]),
    );
    graph.tensors['zp'] = new Tensor('zp', [1], 'uint8', true, false, new Uint8Array([128]));

    graph.nodes.push(
      new Node(
        'QuantizeLinear',
        ['X', 'scale', 'zp'],
        ['X_quant'],
        {
          axis: { name: 'axis', type: 'INT', value: 1 } as Object,
        },
        'q1',
      ),
    );

    // Int16
    graph.tensors['scale2'] = new Tensor(
      'scale2',
      [1],
      'float32',
      true,
      false,
      new Float32Array([0.5]),
    );
    graph.tensors['zp2'] = new Tensor('zp2', [1], 'int16', true, false, new Int16Array([128]));
    graph.nodes.push(
      new Node(
        'QuantizeLinear',
        ['X2', 'scale2', 'zp2'],
        ['X_quant2'],
        { axis: { name: 'axis', type: 'INT', value: 1 } as Object },
        'q2',
      ),
    );

    // Int8
    graph.tensors['scale3'] = new Tensor(
      'scale3',
      [1],
      'float32',
      true,
      false,
      new Float32Array([0.5]),
    );
    graph.tensors['zp3'] = new Tensor('zp3', [1], 'int8', true, false, new Int8Array([-5]));
    graph.nodes.push(
      new Node(
        'QuantizeLinear',
        ['X3', 'scale3', 'zp3'],
        ['X_quant3'],
        { axis: { name: 'axis', type: 'INT', value: 1 } as Object },
        'q3',
      ),
    );

    // Per-channel Dynamic
    graph.tensors['scale4'] = new Tensor(
      'scale4',
      [2],
      'float32',
      true,
      false,
      new Float32Array([0.5, 0.5]),
    );
    graph.tensors['zp4'] = new Tensor('zp4', [2], 'uint8', true, false, new Uint8Array([128, 128]));
    graph.nodes.push(
      new Node(
        'DynamicQuantizeLinear',
        ['X4', 'scale4', 'zp4'],
        ['X_quant4'],
        { axis: { name: 'axis', type: 'INT', value: 1 } as Object },
        'q4',
      ),
    );

    // Simulate a fused conv
    graph.nodes.push(
      new Node(
        'Conv',
        ['X_quant', 'W_quant'],
        ['Y_quant'],
        {
          fused_activation: { name: 'fused_activation', type: 'STRING', value: 'Relu6' } as Object,
        },
        'conv_relu6',
      ),
    );
    graph.nodes.push(
      new Node(
        'Conv',
        ['X_quant2', 'W_quant'],
        ['Y_quant2'],
        {
          fused_activation: { name: 'fused_activation', type: 'STRING', value: 'Relu' } as Object,
        },
        'conv_relu',
      ),
    );
    graph.nodes.push(new Node('Conv', ['X_quant3', 'W_quant'], ['Y_quant3'], {}, 'conv_none'));

    const quantizer = new Quantizer(graph, { mode: 'int8' });
    quantizer.quantize();

    // Verify extraction mapped back explicitly
    const offset1 = quantizer.getQuantizationOffset(
      {
        startObject: () => {},
        addFieldOffset: () => {},
        addFieldInt8: () => {},
        addFieldInt32: () => {},
        endObject: () => 42,
        createFloat32Vector: () => 1,
        createInt64Vector: () => 2,
        startVector: () => {},
        addFloat32: () => {},
        writeInt32: () => {},
        endVector: () => 3,
      } as Object,
      graph.tensors['X_quant'] || new Tensor('X_quant', [1], 'int8'),
    );

    expect(offset1).toBeGreaterThan(0);
  });

  it('should warn on int8', () => {
    const graph = new Graph('TestGraph');
    const quantizer = new Quantizer(graph, { mode: 'int8' });
    quantizer.quantize(); // Warns, doesn't throw
  });

  it('should do nothing on none', () => {
    const graph = new Graph('TestGraph');
    graph.tensors['W'] = new Tensor('W', [1], 'float32', true, false, new Float32Array([1.0]));
    const quantizer = new Quantizer(graph, { mode: 'none' });
    quantizer.quantize();
    expect(graph.tensors['W'].dtype).toBe('float32');
  });
});
