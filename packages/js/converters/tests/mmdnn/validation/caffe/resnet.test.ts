import { describe, it, expect } from 'vitest';
import { parsePrototxt } from '../../../../src/mmdnn/caffe/parser.js';
import { CaffeMapper } from '../../../../src/mmdnn/caffe/mapper.js';
import { Graph } from '@onnx9000/core';

describe('Caffe ResNet-50 Parity', () => {
  it('should map ResNet-50 shortcut blocks to ONNX', () => {
    const prototxt = `

name: "ResNet-50"
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
 
name: "bn1" 
type: "BatchNorm" 
bottom: "conv1" 
top: "bn1" 
}
layer {
 
name: "scale1" 
type: "Scale" 
bottom: "bn1" 
top: "scale1" 
}
layer {
 
name: "relu1" 
type: "ReLU" 
bottom: "scale1" 
top: "scale1" 
}
layer {
 
name: "pool1" 
type: "Pooling" 
bottom: "scale1" 
top: "pool1" 
pooling_param {
 pool: 0 
} 
}
layer {
 
name: "res2a_branch1" 
type: "Convolution" 
bottom: "pool1" 
top: "res2a_branch1" 
}
layer {
 
name: "res2a_branch2a" 
type: "Convolution" 
bottom: "pool1" 
top: "res2a_branch2a" 
}
layer {
 
name: "res2a_branch2b" 
type: "Convolution" 
bottom: "res2a_branch2a" 
top: "res2a_branch2b" 
}
layer {
 
name: "res2a_branch2c" 
type: "Convolution" 
bottom: "res2a_branch2b" 
top: "res2a_branch2c" 
}
layer {
 
name: "res2a" 
type: "Eltwise" 
bottom: "res2a_branch1" 
bottom: "res2a_branch2c" 
top: "res2a" 
eltwise_param {
 operation: 1 
} 
}
        `;

    const parser = { parsePrototxt };
    const model = parser.parsePrototxt(prototxt);
    const mapper = new CaffeMapper();
    const graph = new Graph('resnet');
    for (const layer of model.layer) {
      mapper.map(layer, graph).forEach((n) => graph.addNode(n));
    }

    expect(graph.nodes.some((n) => n.opType === 'Conv')).toBe(true);
    expect(graph.nodes.some((n) => n.opType === 'BatchNormalization')).toBe(true);
    expect(graph.nodes.some((n) => n.opType === 'Add')).toBe(true);
  });
});
