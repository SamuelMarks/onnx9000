import { describe, it, expect } from 'vitest';
import { parseCfg } from '../../../../src/mmdnn/darknet/parser.js';
import { DarknetMapper } from '../../../../src/mmdnn/darknet/mapper.js';
import { Graph } from '@onnx9000/core';

describe('Darknet YOLO v2 Validation', () => {
  it('should parse and map a dummy YOLO v2 cfg', () => {
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

[maxpool]
size=2
stride=2

[route]
layers=-1

[region]
anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
classes=80
coords=4
num=5
`;
    const layers = parseCfg(cfg);
    expect(layers.length).toBe(5);

    const graph = new Graph('yolov2');
    const weights = new Float32Array(1000);
    const mapper = new DarknetMapper(graph, weights);

    expect(() => mapper.map(layers)).not.toThrow();

    const regionLayer = layers.find((l) => l.type === 'region');
    expect(regionLayer).toBeDefined();
    expect(regionLayer?.anchors).toEqual([
      1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52,
    ]);

    // Check that graph mapped correctly
    expect(graph.nodes.some((n) => n.opType === 'Conv')).toBe(true);
    expect(graph.nodes.some((n) => n.opType === 'MaxPool')).toBe(true);
    expect(graph.nodes.some((n) => n.opType === 'Identity')).toBe(true); // Single route
  });
});
