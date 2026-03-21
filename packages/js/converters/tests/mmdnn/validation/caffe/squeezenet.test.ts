import { describe, it, expect } from 'vitest';
import { parsePrototxt } from '../../../../src/mmdnn/caffe/parser.js';
import { CaffeMapper } from '../../../../src/mmdnn/caffe/mapper.js';
import { Graph } from '@onnx9000/core';

describe('Caffe SqueezeNet Parity', () => {
  it('should map SqueezeNet Fire blocks to ONNX', () => {
    const prototxt = `

name: "SqueezeNet"
layer {
 
name: "data" 
type: "Input" 
top: "data" 
}
layer {
 
name: "conv1" 
type: "Convolution" 
bottom: "data" 
top: "conv1" 
}
layer {
 
name: "relu_conv1" 
type: "ReLU" 
bottom: "conv1" 
top: "conv1" 
}
layer {
 
name: "pool1" 
type: "Pooling" 
bottom: "conv1" 
top: "pool1" 
pooling_param {
 pool: 0 
} 
}
layer {
 
name: "fire2/squeeze1x1" 
type: "Convolution" 
bottom: "pool1" 
top: "fire2/squeeze1x1" 
}
layer {
 
name: "fire2/relu_squeeze1x1" 
type: "ReLU" 
bottom: "fire2/squeeze1x1" 
top: "fire2/squeeze1x1" 
}
layer {
 
name: "fire2/expand1x1" 
type: "Convolution" 
bottom: "fire2/squeeze1x1" 
top: "fire2/expand1x1" 
}
layer {
 
name: "fire2/relu_expand1x1" 
type: "ReLU" 
bottom: "fire2/expand1x1" 
top: "fire2/expand1x1" 
}
layer {
 
name: "fire2/expand3x3" 
type: "Convolution" 
bottom: "fire2/squeeze1x1" 
top: "fire2/expand3x3" 
}
layer {
 
name: "fire2/relu_expand3x3" 
type: "ReLU" 
bottom: "fire2/expand3x3" 
top: "fire2/expand3x3" 
}
layer {
 
name: "fire2/concat" 
type: "Concat" 
bottom: "fire2/expand1x1" 
bottom: "fire2/expand3x3" 
top: "fire2/concat" 
}
layer {
 
name: "drop9" 
type: "Dropout" 
bottom: "fire9/concat" 
top: "fire9/concat" 
}
layer {
 
name: "conv10" 
type: "Convolution" 
bottom: "fire9/concat" 
top: "conv10" 
}
layer {
 
name: "relu_conv10" 
type: "ReLU" 
bottom: "conv10" 
top: "conv10" 
}
layer {
 
name: "pool10" 
type: "Pooling" 
bottom: "conv10" 
top: "pool10" 
pooling_param {
 pool: 1 
} 
}
        `;

    const parser = { parsePrototxt };
    const model = parser.parsePrototxt(prototxt);
    const mapper = new CaffeMapper();
    const graph = new Graph('squeezenet');
    for (const layer of model.layer) {
      mapper.map(layer, graph).forEach((n) => graph.addNode(n));
    }

    expect(graph.nodes.some((n) => n.opType === 'Conv')).toBe(true);
    expect(graph.nodes.some((n) => n.opType === 'Concat')).toBe(true);
    // Dropout is usually mapped to Identity or dropped
    expect(graph.nodes.some((n) => n.opType === 'AveragePool')).toBe(true);
  });
});
