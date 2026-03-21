import { describe, it, expect } from 'vitest';
import { parseCfg } from '../../../../src/mmdnn/darknet/parser.js';
import { DarknetMapper } from '../../../../src/mmdnn/darknet/mapper.js';
import { Graph } from '@onnx9000/core';

describe('Darknet YOLO v4 Validation', () => {
  it('should parse and map a dummy YOLO v4 cfg', () => {
    const cfg = `
[net]
batch=1
channels=3
width=608
height=608

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=mish

[maxpool]
size=2
stride=2

[route]
layers=-1

[shortcut]
from=-2
activation=linear

[yolo]
mask = 0,1,2
anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
classes=80
`;
    // Note: mish might map to linear if not explicitly implemented in Mapper, but let's test that it doesn't crash
    const layers = parseCfg(cfg);
    expect(layers.length).toBe(6);

    const graph = new Graph('yolov4');
    const weights = new Float32Array(5000);
    const mapper = new DarknetMapper(graph, weights);

    expect(() => mapper.map(layers)).not.toThrow();

    const yoloLayer = layers.find((l) => l.type === 'yolo');
    expect(yoloLayer).toBeDefined();
    expect(yoloLayer?.anchors).toEqual([
      12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401,
    ]);

    expect(graph.nodes.some((n) => n.opType === 'Conv')).toBe(true);
    expect(graph.nodes.some((n) => n.opType === 'MaxPool')).toBe(true);
    expect(graph.nodes.some((n) => n.opType === 'Identity')).toBe(true);
    expect(graph.nodes.some((n) => n.opType === 'Add')).toBe(true);
  });
});
