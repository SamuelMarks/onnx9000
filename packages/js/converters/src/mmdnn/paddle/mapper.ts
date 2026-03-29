/* eslint-disable */
// @ts-nocheck
import { Graph, Node, Attribute, Tensor, Shape, DynamicDim } from '@onnx9000/core';

type MapperFn = (layer: object, graph: Graph) => Node[];

const paddleRegistry: Record<string, MapperFn> = {};

export function register_paddle_op(domain: string, opType: string) {
  return function (target: object, propertyKey: string, descriptor: PropertyDescriptor) {
    if (domain === 'paddle') {
      paddleRegistry[opType] = descriptor.value.bind(target);
    }
  };
}

// Translate Paddle dynamic lod_tensor shapes to ONNX dynamic axes correctly.
export function translatePaddleShape(paddleShape: number[]): Shape {
  return paddleShape.map((dim) => (dim === -1 ? 'dynamic' : dim)) as Shape;
}

export class PaddleMapper {
  map(layer: object, graph: Graph): Node[] {
    const type = layer.type;
    if (paddleRegistry[type]) {
      return paddleRegistry[type](layer, graph);
    }
    console.warn(`Unsupported Paddle layer type: ${type}`);
    return [];
  }

  @register_paddle_op('paddle', 'conv2d')
  mapConv2d(layer: object, graph: Graph): Node[] {
    const attrs = layer.attrs || {};
    const pads = attrs.paddings || [0, 0, 0, 0];
    const strides = attrs.strides || [1, 1];
    const dilations = attrs.dilations || [1, 1];
    const groups = attrs.groups || 1;

    // Paddle inputs typically dict of lists
    // inputs: { Input: ['x'], Filter: ['w'] }
    const inputs = [];
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

  @register_paddle_op('paddle', 'pool2d')
  mapPool2d(layer: object, graph: Graph): Node[] {
    const attrs = layer.attrs || {};
    const pads = attrs.paddings || [0, 0, 0, 0];
    const strides = attrs.strides || [1, 1];
    const kernel_shape = attrs.ksize || [1, 1];

    // pooling_type: 'max' or 'avg'
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

  @register_paddle_op('paddle', 'elementwise_add')
  mapElementwiseAdd(layer: object, graph: Graph): Node[] {
    const inputs = [];
    if (layer.inputs) {
      if (layer.inputs.X) inputs.push(...layer.inputs.X);
      if (layer.inputs.Y) inputs.push(...layer.inputs.Y);
    }
    const outputs = layer.outputs?.Out || [layer.name || 'add_out'];

    // In ONNX, Add supports broadcasting by default in recent opset versions.
    const node = new Node('Add', inputs, outputs, {}, layer.name || 'elementwise_add');
    return [node];
  }

  @register_paddle_op('paddle', 'relu')
  mapRelu(layer: object, graph: Graph): Node[] {
    const inputs = layer.inputs?.X || [];
    const outputs = layer.outputs?.Out || [layer.name || 'relu_out'];
    const node = new Node('Relu', inputs, outputs, {}, layer.name || 'relu');
    return [node];
  }

  @register_paddle_op('paddle', 'batch_norm')
  mapBatchNorm(layer: object, graph: Graph): Node[] {
    const attrs = layer.attrs || {};
    const epsilon = attrs.epsilon || 1e-5;
    const momentum = attrs.momentum || 0.9;

    const inputs = [];
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

  @register_paddle_op('paddle', 'mul')
  mapMul(layer: object, graph: Graph): Node[] {
    // Paddle's 'mul' is matrix multiplication, often flattened first if rank > 2.
    // X * Y
    const attrs = layer.attrs || {};
    const x_num_col_dims = attrs.x_num_col_dims || 1;
    const y_num_col_dims = attrs.y_num_col_dims || 1;

    const inputs = [];
    if (layer.inputs) {
      if (layer.inputs.X) inputs.push(...layer.inputs.X);
      if (layer.inputs.Y) inputs.push(...layer.inputs.Y);
    }
    const outputs = layer.outputs?.Out || [layer.name || 'mul_out'];

    // If flattening was involved in Paddle, mapping correctly to MatMul or Gemm
    // For MMDNN direct translation, 'mul' -> 'MatMul' typically suffices, assuming standard inputs
    const node = new Node('MatMul', inputs, outputs, {}, layer.name || 'mul');
    return [node];
  }

  @register_paddle_op('paddle', 'concat')
  mapConcat(layer: object, graph: Graph): Node[] {
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

  @register_paddle_op('paddle', 'split')
  mapSplit(layer: object, graph: Graph): Node[] {
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

  @register_paddle_op('paddle', 'matmul')
  mapMatMul(layer: object, graph: Graph): Node[] {
    const inputsX = layer.inputs?.X || [];
    const inputsY = layer.inputs?.Y || [];
    const outputs = layer.outputs?.Out || [layer.name || 'matmul_out'];

    const nodes: Node[] = [];
    let xName = inputsX[0];
    let yName = inputsY[0];

    const attrs = layer.attrs || {};
    if (attrs.transpose_x || attrs.transpose_X) {
      const transXName = xName + '_trans';
      nodes.push(new Node('Transpose', [xName], [transXName], {}, transXName));
      xName = transXName;
    }
    if (attrs.transpose_y || attrs.transpose_Y) {
      const transYName = yName + '_trans';
      nodes.push(new Node('Transpose', [yName], [transYName], {}, transYName));
      yName = transYName;
    }

    const nodeAttrs: Record<string, Attribute> = {};
    const node = new Node('MatMul', [xName, yName], outputs, nodeAttrs, layer.name || 'matmul');
    nodes.push(node);

    return nodes;
  }
}
