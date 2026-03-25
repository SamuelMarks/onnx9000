import { describe, it, expect } from 'vitest';
import { Graph, Node } from '@onnx9000/core';
import { TensorFlowGenerator } from '../../src/mmdnn/tensorflow/generator.js';
import { CaffeGenerator } from '../../src/mmdnn/caffe/generator.js';
import { MXNetGenerator } from '../../src/mmdnn/mxnet/generator.js';
import { CNTKGenerator } from '../../src/mmdnn/cntk/generator.js';
import { KerasGenerator } from '../../src/mmdnn/keras/generator.js';

describe('Fallback Generators', () => {
  const createMockGraph = () => {
    const graph = new Graph('TestGraph');
    graph.inputs = [{ name: 'input_1', type: 'tensor', shape: [1, 3, 224, 224] }];
    graph.outputs = [{ name: 'output_1', type: 'tensor', shape: [1, 1000] }];
    graph.valueInfo = [{ name: 'weight_1', type: 'tensor', shape: [32, 3, 3, 3] }];

    const convNode = new Node(
      'Conv',
      ['input_1', 'weight_1'],
      ['conv_out'],
      { strides: [1, 1], pads: [1, 1, 1, 1] },
      'conv1',
    );
    const reluNode = new Node('Relu', ['conv_out'], ['relu_out'], {}, 'relu1');
    const poolNode = new Node(
      'MaxPool',
      ['relu_out'],
      ['pool_out'],
      { kernel_shape: [2, 2], strides: [2, 2], pads: [0, 0, 0, 0] },
      'pool1',
    );
    const flattenNode = new Node('Flatten', ['pool_out'], ['flat_out'], {}, 'flatten1');
    const denseNode = new Node(
      'Gemm',
      ['flat_out', 'weight_1'],
      ['dense_out'],
      { transB: 1 },
      'dense1',
    );
    const softmaxNode = new Node('Softmax', ['dense_out'], ['output_1'], { axis: 1 }, 'softmax1');

    // Add GAP and Add just to cover those paths
    const gapNode = new Node('GlobalAveragePool', ['conv_out'], ['gap_out'], {}, 'gap1');
    const addNode = new Node('Add', ['gap_out', 'conv_out'], ['add_out'], {}, 'add1');
    const avgPoolNode = new Node(
      'AveragePool',
      ['add_out'],
      ['avg_out'],
      { kernel_shape: [2, 2], strides: [2, 2], pads: [1, 1, 1, 1] },
      'avgpool1',
    );
    const matmulNode = new Node(
      'MatMul',
      ['avg_out', 'weight_1'],
      ['matmul_out'],
      { transB: 1 },
      'matmul1',
    );
    const unknownNode = new Node('UnknownOp', ['matmul_out'], ['unknown_out'], {}, 'unknown1');

    graph.nodes.push(
      convNode,
      reluNode,
      poolNode,
      flattenNode,
      denseNode,
      softmaxNode,
      gapNode,
      addNode,
      avgPoolNode,
      matmulNode,
      unknownNode,
    );
    return graph;
  };

  it('should generate TensorFlow Python code', () => {
    const graph = createMockGraph();
    const generator = new TensorFlowGenerator(graph);
    const code = generator.generate();

    expect(code).toContain('import tensorflow as tf');
    expect(code).toContain('class TestGraph(keras.Model):');
    expect(code).toContain('self.conv_conv1 = layers.Conv2D(');
    expect(code).toContain('self.pool_pool1 = layers.MaxPooling2D(');
    expect(code).toContain('self.flatten_flatten1 = layers.Flatten()');
    expect(code).toContain('self.dense_dense1 = layers.Dense(');
    expect(code).toContain('tf.nn.softmax(');
    expect(code).toContain('self.gap_gap1 = layers.GlobalAveragePooling2D()');
    expect(code).toContain('tf.add(');
    expect(code).toContain('self.pool_avgpool1 = layers.AveragePooling2D(');
    expect(code).toContain('tf.identity(matmul_out)  # Fallback for UnknownOp');
  });

  it('should generate Caffe prototxt code', () => {
    const graph = createMockGraph();
    const generator = new CaffeGenerator(graph);
    const code = generator.generate();

    expect(code).toContain('name: "TestGraph"');
    expect(code).toContain('type: "Input"');
    expect(code).toContain('type: "Convolution"');
    expect(code).toContain('type: "Pooling"');
    expect(code).toContain('type: "ReLU"');
    expect(code).toContain('type: "Flatten"');
    expect(code).toContain('type: "InnerProduct"');
    expect(code).toContain('type: "Softmax"');
    expect(code).toContain('global_pooling: true');
    expect(code).toContain('type: "Eltwise"');
    expect(code).toContain('type: "DummyUnknownOp"');
  });

  it('should generate MXNet Python code', () => {
    const graph = createMockGraph();
    const generator = new MXNetGenerator(graph);
    const code = generator.generate();

    expect(code).toContain('import mxnet as mx');
    expect(code).toContain('class TestGraph(gluon.HybridBlock):');
    expect(code).toContain('self.conv_conv1 = gluon.nn.Conv2D(');
    expect(code).toContain('self.pool_pool1 = gluon.nn.MaxPool2D(');
    expect(code).toContain('self.flatten_flatten1 = gluon.nn.Flatten()');
    expect(code).toContain('self.dense_dense1 = gluon.nn.Dense(');
    expect(code).toContain('mx.nd.softmax(');
    expect(code).toContain('self.gap_gap1 = gluon.nn.GlobalAvgPool2D()');
    expect(code).toContain('mx.nd.add(');
    expect(code).toContain('self.pool_avgpool1 = gluon.nn.AvgPool2D(');
    expect(code).toContain('matmul_out  # Fallback for UnknownOp');
  });

  it('should generate CNTK Python code', () => {
    const graph = createMockGraph();
    const generator = new CNTKGenerator(graph);
    const code = generator.generate();

    expect(code).toContain('import cntk as C');
    expect(code).toContain('def create_TestGraph(inputs):');
    expect(code).toContain('C.layers.Convolution2D(');
    expect(code).toContain('C.layers.MaxPooling(');
    expect(code).toContain('C.flatten(');
    expect(code).toContain('C.layers.Dense(');
    expect(code).toContain('C.softmax(');
    expect(code).toContain('C.layers.GlobalAveragePooling()');
    expect(code).toContain('C.plus(');
    expect(code).toContain('C.layers.AveragePooling(');
    expect(code).toContain('matmul_out  # Fallback for UnknownOp');
  });

  it('should generate Keras Python code', () => {
    const graph = createMockGraph();
    const generator = new KerasGenerator(graph);
    const code = generator.generate();

    expect(code).toContain('import keras');
    expect(code).toContain('def create_TestGraph():');
    expect(code).toContain('Conv2D(');
    expect(code).toContain('MaxPooling2D(');
    expect(code).toContain('Flatten(');
    expect(code).toContain('Dense(');
    expect(code).toContain("Activation('softmax'");
    expect(code).toContain('GlobalAveragePooling2D(');
    expect(code).toContain('Add(');
    expect(code).toContain('AveragePooling2D(');
    expect(code).toContain('matmul_out  # Fallback for UnknownOp');
  });

  it('should handle empty graph gracefully', () => {
    const graph = new Graph('');

    const tfCode = new TensorFlowGenerator(graph).generate();
    expect(tfCode).toContain('pass');

    const mxCode = new MXNetGenerator(graph).generate();
    expect(mxCode).toContain('pass');

    const cntkCode = new CNTKGenerator(graph).generate();
    expect(cntkCode).toContain('pass');

    const kerasCode = new KerasGenerator(graph).generate();
    expect(kerasCode).toContain('pass');
  });

  it('should hit edge cases for code generation (sanitize, getShape, etc)', () => {
    const graph = new Graph('EdgeGraph');
    graph.inputs = [{ name: '0_input', type: 'tensor', shape: [1] }];
    graph.tensors['tensor_1'] = {
      name: 'tensor_1',
      type: 'tensor',
      shape: [2, 2],
      dtype: 'float32',
      data: new Uint8Array(),
    };

    const edgeNode = new Node(
      'Relu',
      ['0_input', 'tensor_1', 'missing_info'],
      ['1_output', '2_output'],
      {},
      '',
    );
    graph.nodes.push(edgeNode);

    const kerasCode = new KerasGenerator(graph).generate();
    expect(kerasCode).toContain('v_0_input');
    expect(kerasCode).toContain('v_1_output');

    const tfCode = new TensorFlowGenerator(graph).generate();
    expect(tfCode).toContain('v_0_input');

    const cntkCode = new CNTKGenerator(graph).generate();
    expect(cntkCode).toContain('v_0_input');
  });
});
