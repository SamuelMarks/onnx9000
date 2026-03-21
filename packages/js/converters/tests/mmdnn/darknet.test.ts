import { describe, it, expect } from 'vitest';
import { parseCfg, parseWeights } from '../../src/mmdnn/darknet/parser.js';
import { DarknetMapper } from '../../src/mmdnn/darknet/mapper.js';
import { Graph } from '@onnx9000/core';

describe('MMDNN - Darknet Parser', () => {
  it('should parse .cfg correctly', () => {
    const cfg = `
# Comment line
[net]
batch=1
subdivisions=1
width=416
height=416
channels=3

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky
string_val=test

[maxpool]
size=2
stride=2

[shortcut]
from=-3
activation=linear

[route]
layers=-1, -4

[yolo]
mask = 0,1,2
anchors = 10,14, 23,27
classes=80
`;
    const layers = parseCfg(cfg);
    expect(layers.length).toBe(6);
    expect(layers[0]!.type).toBe('net');
    expect(layers[0]!.width).toBe(416);
    expect(layers[1]!.type).toBe('convolutional');
    expect(layers[1]!.batch_normalize).toBe(1);
    expect(layers[1]!.filters).toBe(32);
    expect(layers[1]!.string_val).toBe('test');
    expect(layers[2]!.type).toBe('maxpool');
    expect(layers[3]!.type).toBe('shortcut');
    expect(layers[4]!.type).toBe('route');
    expect(layers[4]!.layers).toEqual([-1, -4]);
    expect(layers[5]!.type).toBe('yolo');
    expect(layers[5]!.anchors).toEqual([10, 14, 23, 27]);
  });

  it('should parse weights correctly', () => {
    // Too small buffer
    const smallBuffer = new ArrayBuffer(8);
    let weights = parseWeights(smallBuffer);
    expect(weights.length).toBe(0);

    // v1 version (major=0, minor=1)
    const v1Buffer = new ArrayBuffer(24);
    const view1 = new DataView(v1Buffer);
    view1.setInt32(0, 0, true); // major
    view1.setInt32(4, 1, true); // minor
    view1.setInt32(8, 0, true); // revision
    view1.setInt32(12, 123, true); // seen (4 bytes)
    // weights at 16
    const floatView1 = new Float32Array(v1Buffer, 16);
    floatView1[0] = 1.5;
    floatView1[1] = 2.5;

    weights = parseWeights(v1Buffer);
    expect(weights.length).toBe(2);
    expect(weights[0]).toBeCloseTo(1.5);
    expect(weights[1]).toBeCloseTo(2.5);

    // v2+ version (major=0, minor=2) -> offset should be 20
    const v2Buffer = new ArrayBuffer(28);
    const view2 = new DataView(v2Buffer);
    view2.setInt32(0, 0, true); // major
    view2.setInt32(4, 2, true); // minor
    view2.setInt32(8, 0, true); // revision
    // seen is 8 bytes
    view2.setInt32(12, 123, true); // seen lower
    view2.setInt32(16, 0, true); // seen upper
    // weights at 20
    const floatView2 = new Float32Array(v2Buffer, 20);
    floatView2[0] = 3.5;
    floatView2[1] = 4.5;

    weights = parseWeights(v2Buffer);
    expect(weights.length).toBe(2);
    expect(weights[0]).toBeCloseTo(3.5);
    expect(weights[1]).toBeCloseTo(4.5);

    // Test truncation
    const truncatedBuffer = new ArrayBuffer(20);
    const view3 = new DataView(truncatedBuffer);
    view3.setInt32(0, 0, true); // major
    view3.setInt32(4, 2, true); // minor
    // buffer size is 20, so offset is capped to 20
    weights = parseWeights(truncatedBuffer);
    expect(weights.length).toBe(0);
  });
});

describe('MMDNN - Darknet Mapper', () => {
  it('should map darknet layers to ONNX nodes', () => {
    const cfg = `
[net]
batch=1
channels=3
width=416
height=416

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=mish

[convolutional]
filters=16
size=1
stride=1
activation=swish

[maxpool]
size=2
stride=2

[avgpool]

[connected]
output=1000
activation=linear

[upsample]
stride=2

[route]
layers=-1

[route]
layers=-1, -4

[shortcut]
from=-3
activation=linear

[shortcut]
from=1
activation=linear

[yolo]
classes=80

[region]
classes=20
`;
    const layers = parseCfg(cfg);
    const graph = new Graph('darknet');

    // We need enough weights for all these layers to avoid crash
    // First conv: bn(4*32) + weights(32*3*3*3) = 128 + 864 = 992
    // Second conv: bias(16) + weights(16*32*1*1) = 16 + 512 = 528
    // Connected: bias(1000) + weights(1000*16) = 1000 + 16000 = 17000
    // Total needed ~20000
    const weights = new Float32Array(20000);
    weights.fill(0.1);

    const mapper = new DarknetMapper(graph, weights);
    mapper.map(layers);

    const nodes = graph.nodes;

    // Check conv batch_norm mish
    expect(nodes.some((n) => n.opType === 'Conv' && n.inputs.length === 2)).toBe(true);
    expect(nodes.some((n) => n.opType === 'BatchNormalization')).toBe(true);
    expect(nodes.some((n) => n.opType === 'Mish')).toBe(true);

    // Check conv no bn swish
    expect(nodes.some((n) => n.opType === 'Conv' && n.inputs.length === 3)).toBe(true);
    expect(nodes.some((n) => n.opType === 'Sigmoid')).toBe(true);
    expect(nodes.some((n) => n.opType === 'Mul')).toBe(true);

    // Check pools
    expect(nodes.some((n) => n.opType === 'MaxPool')).toBe(true);
    expect(nodes.some((n) => n.opType === 'AveragePool')).toBe(true);

    // Check gemm
    expect(nodes.some((n) => n.opType === 'Gemm')).toBe(true);

    // Check upsample (Resize)
    expect(nodes.some((n) => n.opType === 'Resize')).toBe(true);

    // Check route (Identity for 1 input, Concat for >1)
    expect(nodes.some((n) => n.opType === 'Identity')).toBe(true);
    expect(nodes.some((n) => n.opType === 'Concat')).toBe(true);

    // Check shortcut (Add)
    expect(nodes.some((n) => n.opType === 'Add')).toBe(true);
  });

  it('should handle edge cases with missing properties or missing weights', () => {
    const cfg = `
[net]

[convolutional]
activation=linear

[maxpool]

[connected]

[route]
layers=0

[upsample]
`;
    const layers = parseCfg(cfg);
    const graph = new Graph('darknet2');

    // Insufficient weights
    const weights = new Float32Array(2);
    const mapper = new DarknetMapper(graph, weights);
    mapper.map(layers);

    const nodes = graph.nodes;
    expect(nodes.some((n) => n.opType === 'Conv')).toBe(true);
    expect(nodes.some((n) => n.opType === 'MaxPool')).toBe(true);
    expect(nodes.some((n) => n.opType === 'Gemm')).toBe(true);
    expect(nodes.some((n) => n.opType === 'Identity')).toBe(true);
    expect(nodes.some((n) => n.opType === 'Resize')).toBe(true);
  });

  it('should map without net layer', () => {
    const layers = [{ type: 'convolutional', filters: 3, size: 1 }];
    const graph = new Graph('no_net');
    const mapper = new DarknetMapper(graph, new Float32Array(100));
    mapper.map(layers);
    expect(graph.inputs[0]?.shape).toEqual(['batch_size', 3, 416, 416]);
    expect(graph.outputs[0]?.shape).toEqual(['batch_size', -1, -1, -1]);
  });

  it('should map unknown layer to nothing', () => {
    const layers = [{ type: 'net' }, { type: 'unknown_layer' }];
    const graph = new Graph('unknown');
    const mapper = new DarknetMapper(graph, new Float32Array(100));
    mapper.map(layers);
    expect(graph.nodes.length).toBe(0);
  });
});
