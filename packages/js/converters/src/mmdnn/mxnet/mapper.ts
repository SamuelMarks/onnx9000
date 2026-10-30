/**
 * @fileoverview MXNet symbolic operator mapper to ONNX IR nodes.
 */
import { Attribute, type Graph, Node } from '@onnx9000/core';
import type { MxNetNode } from '../types.js';

/** Function signature for mapping an MXNet node to one or more ONNX nodes. */
export type MxNetMapperFn = (node: MxNetNode, graph: Graph) => Node[];

/** Registry mapping MXNet operator names to their corresponding mapper functions. */
const mxnetRegistry: Record<string, MxNetMapperFn> = {};

/**
 * Decorator that registers a method on `MxNetMapper` as an MXNet operator handler.
 * @param domain Operator domain (`'mxnet'`).
 * @param opType MXNet operator type string.
 * @returns Method decorator.
 */
export function register_mxnet_op(domain: string, opType: string) {
  return (target: MxNetMapper, _propertyKey: string, descriptor: PropertyDescriptor): void => {
    if (domain === 'mxnet' && typeof descriptor.value === 'function') {
      mxnetRegistry[opType] = (descriptor.value as MxNetMapperFn).bind(target);
    }
  };
}

/**
 * Parses an MXNet tuple string such as `"(3, 3)"` into an array of numbers.
 * @param str Raw tuple string from MXNet node attributes.
 * @returns Parsed numeric values, or an empty array if absent or invalid.
 */
function parseTuple(str: string | undefined): number[] {
  if (!str) return [];
  const s = str.trim();
  if (s.startsWith('(') && s.endsWith(')')) {
    return s
      .slice(1, -1)
      .split(',')
      .map(Number)
      .filter((n) => !Number.isNaN(n));
  }
  return [];
}

/**
 * Maps MXNet symbolic nodes (`MxNetNode`) into ONNX `Node` instances.
 */
export class MxNetMapper {
  /**
   * Maps a single MXNet symbolic node into ONNX nodes.
   * @param node Parsed MXNet symbolic node.
   * @param graph Target ONNX graph.
   * @returns Emitted ONNX nodes.
   */
  map(node: MxNetNode, graph: Graph): Node[] {
    const type = node.op;
    if (type === 'null') {
      return [];
    }
    const handler = mxnetRegistry[type];
    if (handler) {
      return handler(node, graph);
    }
    console.warn(`Unsupported MXNet layer type: ${type}`);
    return [];
  }

  /**
   * Maps an MXNet `Convolution` node to an ONNX `Conv` node.
   * @param node MXNet Convolution node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Conv` node.
   */
  @register_mxnet_op('mxnet', 'Convolution')
  mapConvolution(node: MxNetNode, _graph: Graph): Node[] {
    const attrs = node.attrs || {};
    const n = new Node('Conv', [], [node.name || ''], {}, node.name || '');
    const kernel = parseTuple(attrs.kernel);
    if (kernel.length > 0)
      n.attributes.kernel_shape = new Attribute('kernel_shape', 'INTS', kernel);
    const stride = parseTuple(attrs.stride);
    if (stride.length > 0) n.attributes.strides = new Attribute('strides', 'INTS', stride);
    const pad = parseTuple(attrs.pad);
    if (pad.length > 0) {
      const pads =
        pad.length === 2
          ? [pad[0] as number, pad[1] as number, pad[0] as number, pad[1] as number]
          : pad;
      n.attributes.pads = new Attribute('pads', 'INTS', pads);
    }
    const dilate = parseTuple(attrs.dilate);
    if (dilate.length > 0) n.attributes.dilations = new Attribute('dilations', 'INTS', dilate);
    if (attrs.num_group)
      n.attributes.group = new Attribute('group', 'INT', parseInt(attrs.num_group, 10));

    return [n];
  }

  /**
   * Maps an MXNet `FullyConnected` node to an ONNX `Gemm` node.
   * @param node MXNet FullyConnected node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Gemm` node.
   */
  @register_mxnet_op('mxnet', 'FullyConnected')
  mapFullyConnected(node: MxNetNode, _graph: Graph): Node[] {
    const n = new Node('Gemm', [], [node.name || ''], {}, node.name || '');
    n.attributes.transB = new Attribute('transB', 'INT', 1);
    return [n];
  }

  /**
   * Maps an MXNet `Activation` node to an ONNX activation node (`Relu`, `Sigmoid`, `Tanh`, `Softplus`).
   * @param node MXNet Activation node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX activation node.
   */
  @register_mxnet_op('mxnet', 'Activation')
  mapActivation(node: MxNetNode, _graph: Graph): Node[] {
    const attrs = node.attrs || {};
    const actType = attrs.act_type || 'relu';
    let op = 'Relu';
    if (actType === 'relu') op = 'Relu';
    else if (actType === 'sigmoid') op = 'Sigmoid';
    else if (actType === 'tanh') op = 'Tanh';
    else if (actType === 'softrelu') op = 'Softplus';

    return [new Node(op, [], [node.name || ''], {}, node.name || '')];
  }

  /**
   * Maps an MXNet `Pooling` node to an ONNX pooling node (`MaxPool`, `AveragePool`, `GlobalMaxPool`, `GlobalAveragePool`).
   * @param node MXNet Pooling node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX pooling node.
   */
  @register_mxnet_op('mxnet', 'Pooling')
  mapPooling(node: MxNetNode, _graph: Graph): Node[] {
    const attrs = node.attrs || {};
    const poolType = attrs.pool_type || 'max';
    const isGlobal = attrs.global_pool === 'True';
    let op = 'MaxPool';
    if (poolType === 'max' && isGlobal) op = 'GlobalMaxPool';
    else if (poolType === 'avg' && isGlobal) op = 'GlobalAveragePool';
    else if (poolType === 'avg') op = 'AveragePool';

    const name = node.name || '';
    const n = new Node(op, [], [name], {}, name);

    if (!isGlobal) {
      const kernel = parseTuple(attrs.kernel);
      if (kernel.length > 0)
        n.attributes.kernel_shape = new Attribute('kernel_shape', 'INTS', kernel);
      const stride = parseTuple(attrs.stride);
      if (stride.length > 0) n.attributes.strides = new Attribute('strides', 'INTS', stride);
      const pad = parseTuple(attrs.pad);
      if (pad.length > 0) {
        const pads =
          pad.length === 2
            ? [pad[0] as number, pad[1] as number, pad[0] as number, pad[1] as number]
            : pad;
        n.attributes.pads = new Attribute('pads', 'INTS', pads);
      }
    }
    return [n];
  }

  /**
   * Maps an MXNet `BatchNorm` node to an ONNX `BatchNormalization` node.
   * @param node MXNet BatchNorm node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `BatchNormalization` node.
   */
  @register_mxnet_op('mxnet', 'BatchNorm')
  mapBatchNorm(node: MxNetNode, _graph: Graph): Node[] {
    const attrs = node.attrs || {};
    const n = new Node('BatchNormalization', [], [node.name || ''], {}, node.name || '');
    if (attrs.eps) n.attributes.epsilon = new Attribute('epsilon', 'FLOAT', parseFloat(attrs.eps));
    if (attrs.momentum)
      n.attributes.momentum = new Attribute('momentum', 'FLOAT', parseFloat(attrs.momentum));
    return [n];
  }

  /**
   * Maps an MXNet `Dropout` node to an ONNX `Identity` node for inference.
   * @param node MXNet Dropout node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Identity` node.
   */
  @register_mxnet_op('mxnet', 'Dropout')
  mapDropout(node: MxNetNode, _graph: Graph): Node[] {
    const n = new Node('Identity', [], [node.name || ''], {}, node.name || '');
    return [n];
  }

  /**
   * Maps an MXNet `Flatten` node to an ONNX `Flatten` node.
   * @param node MXNet Flatten node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Flatten` node.
   */
  @register_mxnet_op('mxnet', 'Flatten')
  mapFlatten(node: MxNetNode, _graph: Graph): Node[] {
    const n = new Node('Flatten', [], [node.name || ''], {}, node.name || '');
    n.attributes.axis = new Attribute('axis', 'INT', 1);
    return [n];
  }

  /**
   * Maps an MXNet `Reshape` node to an ONNX `Reshape` node.
   * @param node MXNet Reshape node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Reshape` node.
   */
  @register_mxnet_op('mxnet', 'Reshape')
  mapReshape(node: MxNetNode, _graph: Graph): Node[] {
    const n = new Node('Reshape', [], [node.name || ''], {}, node.name || '');
    return [n];
  }

  /**
   * Maps an MXNet `Concat` node to an ONNX `Concat` node.
   * @param node MXNet Concat node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Concat` node.
   */
  @register_mxnet_op('mxnet', 'Concat')
  mapConcat(node: MxNetNode, _graph: Graph): Node[] {
    const attrs = node.attrs || {};
    const n = new Node('Concat', [], [node.name || ''], {}, node.name || '');
    n.attributes.axis = new Attribute('axis', 'INT', parseInt(attrs.dim || '1', 10));
    return [n];
  }

  /**
   * Maps an MXNet `elemwise_add` node to an ONNX `Add` node.
   * @param node MXNet elemwise_add node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Add` node.
   */
  @register_mxnet_op('mxnet', 'elemwise_add')
  mapElemwiseAdd(node: MxNetNode, _graph: Graph): Node[] {
    return [new Node('Add', [], [node.name || ''], {}, node.name || '')];
  }

  /**
   * Maps an MXNet `elemwise_sub` node to an ONNX `Sub` node.
   * @param node MXNet elemwise_sub node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Sub` node.
   */
  @register_mxnet_op('mxnet', 'elemwise_sub')
  mapElemwiseSub(node: MxNetNode, _graph: Graph): Node[] {
    return [new Node('Sub', [], [node.name || ''], {}, node.name || '')];
  }

  /**
   * Maps an MXNet `elemwise_mul` node to an ONNX `Mul` node.
   * @param node MXNet elemwise_mul node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Mul` node.
   */
  @register_mxnet_op('mxnet', 'elemwise_mul')
  mapElemwiseMul(node: MxNetNode, _graph: Graph): Node[] {
    return [new Node('Mul', [], [node.name || ''], {}, node.name || '')];
  }

  /**
   * Maps an MXNet `broadcast_add` node to an ONNX `Add` node.
   * @param node MXNet broadcast_add node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Add` node.
   */
  @register_mxnet_op('mxnet', 'broadcast_add')
  mapBroadcastAdd(node: MxNetNode, _graph: Graph): Node[] {
    return [new Node('Add', [], [node.name || ''], {}, node.name || '')];
  }

  /**
   * Maps an MXNet `broadcast_mul` node to an ONNX `Mul` node.
   * @param node MXNet broadcast_mul node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Mul` node.
   */
  @register_mxnet_op('mxnet', 'broadcast_mul')
  mapBroadcastMul(node: MxNetNode, _graph: Graph): Node[] {
    return [new Node('Mul', [], [node.name || ''], {}, node.name || '')];
  }

  /**
   * Maps an MXNet `SoftmaxOutput` node to an ONNX `Softmax` node.
   * @param node MXNet SoftmaxOutput node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Softmax` node.
   */
  @register_mxnet_op('mxnet', 'SoftmaxOutput')
  mapSoftmaxOutput(node: MxNetNode, _graph: Graph): Node[] {
    const n = new Node('Softmax', [], [node.name || ''], {}, node.name || '');
    n.attributes.axis = new Attribute('axis', 'INT', 1);
    return [n];
  }

  /**
   * Maps an MXNet `LeakyReLU` node to an ONNX `LeakyRelu`, `Elu`, or `PRelu` node.
   * @param node MXNet LeakyReLU node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the mapped ONNX activation node.
   */
  @register_mxnet_op('mxnet', 'LeakyReLU')
  mapLeakyReLU(node: MxNetNode, _graph: Graph): Node[] {
    const attrs = node.attrs || {};
    const name = node.name || '';
    const actType = attrs.act_type || 'leaky';
    if (actType === 'leaky') {
      const n = new Node('LeakyRelu', [], [name], {}, name);
      if (attrs.slope) {
        n.attributes.alpha = new Attribute('alpha', 'FLOAT', parseFloat(attrs.slope));
      }
      return [n];
    } else if (actType === 'elu') {
      const n = new Node('Elu', [], [name], {}, name);
      if (attrs.slope)
        n.attributes.alpha = new Attribute('alpha', 'FLOAT', parseFloat(attrs.slope));
      return [n];
    } else if (actType === 'prelu') {
      return [new Node('PRelu', [], [name], {}, name)];
    }
    return [new Node('LeakyRelu', [], [name], {}, name)];
  }

  /**
   * Maps an MXNet `UpSampling` node to an ONNX `Resize` node.
   * @param node MXNet UpSampling node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Resize` node.
   */
  @register_mxnet_op('mxnet', 'UpSampling')
  mapUpSampling(node: MxNetNode, _graph: Graph): Node[] {
    const attrs = node.attrs || {};
    const n = new Node('Resize', [], [node.name || ''], {}, node.name || '');
    if (attrs.sample_type === 'nearest') {
      n.attributes.mode = new Attribute('mode', 'STRING', 'nearest');
    } else {
      n.attributes.mode = new Attribute('mode', 'STRING', 'linear');
    }
    return [n];
  }

  /**
   * Maps an MXNet `SliceChannel` node to an ONNX `Split` node.
   * @param node MXNet SliceChannel node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Split` node.
   */
  @register_mxnet_op('mxnet', 'SliceChannel')
  mapSliceChannel(node: MxNetNode, _graph: Graph): Node[] {
    const attrs = node.attrs || {};
    const n = new Node('Split', [], [node.name || ''], {}, node.name || '');
    if (attrs.axis !== undefined) {
      n.attributes.axis = new Attribute('axis', 'INT', parseInt(attrs.axis, 10));
    } else {
      n.attributes.axis = new Attribute('axis', 'INT', 1);
    }
    return [n];
  }

  /**
   * Maps an MXNet `Crop` node to an ONNX `Slice` node.
   * @param node MXNet Crop node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `Slice` node.
   */
  @register_mxnet_op('mxnet', 'Crop')
  mapCrop(node: MxNetNode, _graph: Graph): Node[] {
    const n = new Node('Slice', [], [node.name || ''], {}, node.name || '');
    return [n];
  }

  /**
   * Maps an MXNet `Deconvolution` node to an ONNX `ConvTranspose` node.
   * @param node MXNet Deconvolution node.
   * @param _graph Target ONNX graph.
   * @returns Single-element array containing the ONNX `ConvTranspose` node.
   */
  @register_mxnet_op('mxnet', 'Deconvolution')
  mapDeconvolution(node: MxNetNode, _graph: Graph): Node[] {
    const attrs = node.attrs || {};
    const n = new Node('ConvTranspose', [], [node.name || ''], {}, node.name || '');
    const kernel = parseTuple(attrs.kernel);
    if (kernel.length > 0)
      n.attributes.kernel_shape = new Attribute('kernel_shape', 'INTS', kernel);
    const stride = parseTuple(attrs.stride);
    if (stride.length > 0) n.attributes.strides = new Attribute('strides', 'INTS', stride);
    const pad = parseTuple(attrs.pad);
    if (pad.length > 0) {
      const pads =
        pad.length === 2
          ? [pad[0] as number, pad[1] as number, pad[0] as number, pad[1] as number]
          : pad;
      n.attributes.pads = new Attribute('pads', 'INTS', pads);
    }

    const dilate = parseTuple(attrs.dilate);
    if (dilate.length > 0) n.attributes.dilations = new Attribute('dilations', 'INTS', dilate);
    if (attrs.num_group)
      n.attributes.group = new Attribute('group', 'INT', parseInt(attrs.num_group, 10));
    return [n];
  }
}
