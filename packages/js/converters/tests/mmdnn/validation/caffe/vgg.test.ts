import { describe, it, expect } from 'vitest';
import { CaffeMapper } from '../../../../src/mmdnn/caffe/mapper.js';
import { parsePrototxt } from '../../../../src/mmdnn/caffe/parser.js';
import { Graph } from '@onnx9000/core';

describe('Caffe Validation - VGG16', () => {
  it('should parse and map VGG16 architecture', () => {
    const prototxt = `
name: "VGG16"
layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "data"
  top: "conv1_1"
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu1_1"
  type: "ReLU"
  bottom: "conv1_1"
  top: "conv1_1"
}
layer {
  name: "conv1_2"
  type: "Convolution"
  bottom: "conv1_1"
  top: "conv1_2"
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu1_2"
  type: "ReLU"
  bottom: "conv1_2"
  top: "conv1_2"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1_2"
  top: "pool1"
  pooling_param {
    pool: 0
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
layer {
  name: "drop6"
  type: "Dropout"
  bottom: "fc6"
  top: "fc6"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc8"
  top: "prob"
}
`;

    const parsed = parsePrototxt(prototxt);
    const mapper = new CaffeMapper();
    const graph = new Graph('vgg16');

    for (const layer of parsed.layer) {
      const nodes = mapper.map(layer, graph);
      for (const node of nodes) {
        graph.addNode(node);
      }
    }

    const opTypes = graph.nodes.map((n) => n.opType);
    expect(opTypes).toContain('Conv');
    expect(opTypes).toContain('Relu');
    expect(opTypes).toContain('MaxPool');
    expect(opTypes).toContain('Gemm');
    expect(opTypes).toContain('Softmax');
  });
});
