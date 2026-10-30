/**
 * @fileoverview PaddlePaddle operator mapper to ONNX IR nodes.
 */
import { Attribute, type Graph, Node, type Shape } from '@onnx9000/core';
import type { PaddleOpDesc } from '../types.js';

/** Function signature for mapping a Paddle operator description to ONNX nodes. */
export type PaddleMapperFn = (layer: PaddleOpDesc, graph: Graph) => Node[];

/** Registry mapping Paddle operator types to their corresponding mapper functions. */
const paddleRegistry: Record<string, PaddleMapperFn> = {};

/**
 * Decorator that registers a method on `PaddleMapper` as a Paddle operator handler.
 * @param domain Operator domain (`'paddle'`).
 * @param opType Paddle operator type string.
 * @returns Method decorator.
 */
export function register_paddle_op(domain: string, opType: string) {
  return (target: PaddleMapper, _propertyKey: string, descriptor: PropertyDescriptor): void => {
    if (domain === 'paddle' && typeof descriptor.value === 'function') {
      paddleRegistry[opType] = (descriptor.value as PaddleMapperFn).bind(target);
    }
  };
}

/**
 * Translates Paddle dynamic `lod_tensor` shapes (`-1`) to ONNX dynamic axes (`'dynamic'`).
 * @param paddleShape Array of integer dimensions from Paddle.
 * @returns ONNX `Shape` array with `-1` converted to `'dynamic'`.
 */
export function translatePaddleShape(paddleShape: number[]): Shape {
  return paddleShape.map((dim) => (dim === -1 ? 'dynamic' : dim));
}

/**
 * Maps PaddlePaddle operators (`PaddleOpDesc`) into ONNX `Node` instances.
 */
export class PaddleMapper {
  /**
   * Maps a single Paddle operator description into ONNX nodes.
   * @param layer Paddle operator description.
   * @param graph Target ONNX graph.
   * @returns Array of emitted ONNX nodes.
   */
  map(layer: PaddleOpDesc, graph: Graph): Node[] {
    const type = layer.type;
    const handler = paddleRegistry[type];
    if (handler) {
      return handler(layer, graph);
    }
    console.warn(`Unsupported Paddle layer type: ${type}`);
    return [];
  }

  /**
   * Maps a Paddle `conv2d` operator to an ONNX `Conv` node.
   * @param layer Paddle conv2d operator description.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Conv` node.
   */
  @register_paddle_op('paddle', 'conv2d')
  mapConv2d(layer: PaddleOpDesc, _graph: Graph): Node[] {
    const attrs = layer.attrs || {};
    const pads = attrs.paddings || [0, 0, 0, 0];
    const strides = attrs.strides || [1, 1];
    const dilations = attrs.dilations || [1, 1];
    const groups = attrs.groups || 1;

    const inputs: string[] = [];
    if (layer.inputs) {
      if (layer.inputs.Input) inputs.push(...layer.inputs.Input);
      if (layer.inputs.Filter) inputs.push(...layer.inputs.Filter);
      if (layer.inputs.Bias) inputs.push(...layer.inputs.Bias);
    }

    const outputs = layer.outputs?.Output || [layer.name || 'conv2d_out'];

    const node = new Node(
      'Conv',
      inputs,
      outputs,
      {
        pads: new Attribute('pads', 'INTS', pads),
        strides: new Attribute('strides', 'INTS', strides),
        dilations: new Attribute('dilations', 'INTS', dilations),
        group: new Attribute('group', 'INT', groups),
      },
      layer.name || 'conv2d',
    );
    return [node];
  }

  /**
   * Maps a Paddle `pool2d` operator to an ONNX `MaxPool` or `AveragePool` node.
   * @param layer Paddle pool2d operator description.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX pooling node.
   */
  @register_paddle_op('paddle', 'pool2d')
  mapPool2d(layer: PaddleOpDesc, _graph: Graph): Node[] {
    const attrs = layer.attrs || {};
    const pads = attrs.paddings || [0, 0, 0, 0];
    const strides = attrs.strides || [1, 1];
    const kernel_shape = attrs.ksize || [1, 1];

    const pooling_type = attrs.pooling_type === 'avg' ? 'AveragePool' : 'MaxPool';
    const ceil_mode = attrs.ceil_mode ? 1 : 0;

    const inputs = layer.inputs?.X || [];
    const outputs = layer.outputs?.Out || [layer.name || 'pool2d_out'];

    const nodeAttrs: Record<string, Attribute> = {
      pads: new Attribute('pads', 'INTS', pads),
      strides: new Attribute('strides', 'INTS', strides),
      kernel_shape: new Attribute('kernel_shape', 'INTS', kernel_shape),
      ceil_mode: new Attribute('ceil_mode', 'INT', ceil_mode),
    };

    const node = new Node(pooling_type, inputs, outputs, nodeAttrs, layer.name || pooling_type);
    return [node];
  }

  /**
   * Maps a Paddle `elementwise_add` operator to an ONNX `Add` node.
   * @param layer Paddle elementwise_add operator description.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Add` node.
   */
  @register_paddle_op('paddle', 'elementwise_add')
  mapElementwiseAdd(layer: PaddleOpDesc, _graph: Graph): Node[] {
    const inputs: string[] = [];
    if (layer.inputs) {
      if (layer.inputs.X) inputs.push(...layer.inputs.X);
      if (layer.inputs.Y) inputs.push(...layer.inputs.Y);
    }
    const outputs = layer.outputs?.Out || [layer.name || 'add_out'];

    const node = new Node('Add', inputs, outputs, {}, layer.name || 'elementwise_add');
    return [node];
  }

  /**
   * Maps a Paddle `relu` operator to an ONNX `Relu` node.
   * @param layer Paddle relu operator description.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Relu` node.
   */
  @register_paddle_op('paddle', 'relu')
  mapRelu(layer: PaddleOpDesc, _graph: Graph): Node[] {
    const inputs = layer.inputs?.X || [];
    const outputs = layer.outputs?.Out || [layer.name || 'relu_out'];
    const node = new Node('Relu', inputs, outputs, {}, layer.name || 'relu');
    return [node];
  }

  /**
   * Maps a Paddle `batch_norm` operator to an ONNX `BatchNormalization` node.
   * @param layer Paddle batch_norm operator description.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `BatchNormalization` node.
   */
  @register_paddle_op('paddle', 'batch_norm')
  mapBatchNorm(layer: PaddleOpDesc, _graph: Graph): Node[] {
    const attrs = layer.attrs || {};
    const epsilon = attrs.epsilon || 1e-5;
    const momentum = attrs.momentum || 0.9;

    const inputs: string[] = [];
    if (layer.inputs) {
      if (layer.inputs.X) inputs.push(...layer.inputs.X);
      if (layer.inputs.Scale) inputs.push(...layer.inputs.Scale);
      if (layer.inputs.Bias) inputs.push(...layer.inputs.Bias);
      if (layer.inputs.Mean) inputs.push(...layer.inputs.Mean);
      if (layer.inputs.Variance) inputs.push(...layer.inputs.Variance);
    }
    const outputs = layer.outputs?.Y || [layer.name || 'batch_norm_out'];

    const nodeAttrs: Record<string, Attribute> = {
      epsilon: new Attribute('epsilon', 'FLOAT', epsilon),
      momentum: new Attribute('momentum', 'FLOAT', momentum),
    };

    const node = new Node(
      'BatchNormalization',
      inputs,
      outputs,
      nodeAttrs,
      layer.name || 'batch_norm',
    );
    return [node];
  }

  /**
   * Maps a Paddle `mul` operator to an ONNX `MatMul` node.
   * @param layer Paddle mul operator description.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `MatMul` node.
   */
  @register_paddle_op('paddle', 'mul')
  mapMul(layer: PaddleOpDesc, _graph: Graph): Node[] {
    const inputs: string[] = [];
    if (layer.inputs) {
      if (layer.inputs.X) inputs.push(...layer.inputs.X);
      if (layer.inputs.Y) inputs.push(...layer.inputs.Y);
    }
    const outputs = layer.outputs?.Out || [layer.name || 'mul_out'];

    const node = new Node('MatMul', inputs, outputs, {}, layer.name || 'mul');
    return [node];
  }

  /**
   * Maps a Paddle `concat` operator to an ONNX `Concat` node.
   * @param layer Paddle concat operator description.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Concat` node.
   */
  @register_paddle_op('paddle', 'concat')
  mapConcat(layer: PaddleOpDesc, _graph: Graph): Node[] {
    const attrs = layer.attrs || {};
    const axis = attrs.axis || 0;

    const inputs = layer.inputs?.X || [];
    const outputs = layer.outputs?.Out || [layer.name || 'concat_out'];

    const nodeAttrs: Record<string, Attribute> = {
      axis: new Attribute('axis', 'INT', axis),
    };

    const node = new Node('Concat', inputs, outputs, nodeAttrs, layer.name || 'concat');
    return [node];
  }

  /**
   * Maps a Paddle `split` operator to an ONNX `Split` node.
   * @param layer Paddle split operator description.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Split` node.
   */
  @register_paddle_op('paddle', 'split')
  mapSplit(layer: PaddleOpDesc, _graph: Graph): Node[] {
    const attrs = layer.attrs || {};
    const axis = attrs.axis || 0;

    const inputs = layer.inputs?.X || [];
    const outputs = layer.outputs?.Out || [];

    const nodeAttrs: Record<string, Attribute> = {
      axis: new Attribute('axis', 'INT', axis),
    };

    if (attrs.num_or_sections !== undefined) {
      if (Array.isArray(attrs.num_or_sections)) {
        nodeAttrs.split = new Attribute('split', 'INTS', attrs.num_or_sections);
      }
    }

    const node = new Node('Split', inputs, outputs, nodeAttrs, layer.name || 'split');
    return [node];
  }

  /**
   * Maps a Paddle `matmul` operator to an ONNX `MatMul` node (with optional `Transpose` nodes).
   * @param layer Paddle matmul operator description.
   * @param _graph Target ONNX graph.
   * @returns Array of ONNX `Transpose` and `MatMul` nodes.
   */
  @register_paddle_op('paddle', 'matmul')
  mapMatMul(layer: PaddleOpDesc, _graph: Graph): Node[] {
    const inputsX = layer.inputs?.X || [];
    const inputsY = layer.inputs?.Y || [];
    const outputs = layer.outputs?.Out || [layer.name || 'matmul_out'];

    const nodes: Node[] = [];
    let xName = inputsX[0] as string;
    let yName = inputsY[0] as string;

    const attrs = layer.attrs || {};
    if (attrs.transpose_x || attrs.transpose_X) {
      const transXName = `${xName}_trans`;
      nodes.push(new Node('Transpose', [xName], [transXName], {}, transXName));
      xName = transXName;
    }
    if (attrs.transpose_y || attrs.transpose_Y) {
      const transYName = `${yName}_trans`;
      nodes.push(new Node('Transpose', [yName], [transYName], {}, transYName));
      yName = transYName;
    }

    const nodeAttrs: Record<string, Attribute> = {};
    const node = new Node('MatMul', [xName, yName], outputs, nodeAttrs, layer.name || 'matmul');
    nodes.push(node);

    return nodes;
  }
}
