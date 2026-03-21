import { describe, it, expect } from 'vitest';
import { parseCfg } from '../../../../src/mmdnn/darknet/parser.js';
import { DarknetMapper } from '../../../../src/mmdnn/darknet/mapper.js';
import { Graph } from '@onnx9000/core';

describe('Darknet YOLO v3 Validation', () => {
  it('should parse and map a dummy YOLO v3 cfg', () => {
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
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[shortcut]
from=-2
activation=linear

[upsample]
stride=2

[route]
layers = -1, -4

[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
`;
    const layers = parseCfg(cfg);
    expect(layers.length).toBe(7);

    const graph = new Graph('yolov3');
    const weights = new Float32Array(10000);
    const mapper = new DarknetMapper(graph, weights);

    expect(() => mapper.map(layers)).not.toThrow();

    const yoloLayer = layers.find((l) => l.type === 'yolo');
    expect(yoloLayer).toBeDefined();
    expect(yoloLayer?.anchors).toEqual([
      10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326,
    ]);

    // Check nodes
    expect(graph.nodes.some((n) => n.opType === 'Conv')).toBe(true);
    expect(graph.nodes.some((n) => n.opType === 'Add')).toBe(true); // shortcut
    expect(graph.nodes.some((n) => n.opType === 'Resize')).toBe(true); // upsample
    expect(graph.nodes.some((n) => n.opType === 'Concat')).toBe(true); // dual route
  });
});
