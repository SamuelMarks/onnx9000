/**
 * @fileoverview Caffe layer parameter mapper to ONNX IR nodes.
 */
import { Attribute, type Graph, Node, Tensor } from '@onnx9000/core';
import type { CaffeBlob, CaffeLayerParam, CaffeSpatialParam } from '../types.js';

/**
 * Function signature that maps a Caffe layer parameter to one or more ONNX Nodes.
 */
export type CaffeMapperFn = (layer: CaffeLayerParam, graph: Graph) => Node[];

/**
 * Registry storing mapping functions for supported Caffe layer types.
 */
const caffeRegistry: Record<string, CaffeMapperFn> = {};

/**
 * Decorator that registers a method as the mapping function for a specific Caffe layer type.
 *
 * @param domain Framework domain (e.g., `'caffe'`).
 * @param opType Caffe layer type to register the mapping for (e.g., `'Convolution'`).
 * @returns Method decorator function.
 */
export function register_caffe_op(domain: string, opType: string) {
  return (target: CaffeMapper, _propertyKey: string, descriptor: PropertyDescriptor): void => {
    if (domain === 'caffe' && typeof descriptor.value === 'function') {
      caffeRegistry[opType] = (descriptor.value as CaffeMapperFn).bind(target);
    }
  };
}

/**
 * Resolves padding values from a Caffe layer parameter into `[pad_top, pad_left, pad_bottom, pad_right]`.
 *
 * @param param Layer spatial parameter containing padding information.
 * @returns Array of 4 padding values.
 */
function resolvePadding(param: CaffeSpatialParam | undefined): number[] {
  if (!param) return [0, 0, 0, 0];
  let pad_h = 0;
  let pad_w = 0;
  if (param.pad !== undefined) {
    const p = Array.isArray(param.pad) ? (param.pad[0] ?? 0) : param.pad;
    pad_h = p;
    pad_w = p;
  } else {
    if (param.pad_h !== undefined) pad_h = param.pad_h;
    if (param.pad_w !== undefined) pad_w = param.pad_w;
  }
  return [pad_h, pad_w, pad_h, pad_w];
}

/**
 * Resolves kernel size values from a Caffe layer parameter into `[kernel_h, kernel_w]`.
 *
 * @param param Layer spatial parameter containing kernel size information.
 * @returns Array of 2 kernel size values.
 */
function resolveKernel(param: CaffeSpatialParam | undefined): number[] {
  if (!param) return [1, 1];
  let kh = 1;
  let kw = 1;
  if (param.kernel_size !== undefined) {
    const k = Array.isArray(param.kernel_size) ? (param.kernel_size[0] ?? 1) : param.kernel_size;
    kh = k;
    kw = k;
  } else {
    if (param.kernel_h !== undefined) kh = param.kernel_h;
    if (param.kernel_w !== undefined) kw = param.kernel_w;
  }
  return [kh, kw];
}

/**
 * Resolves stride values from a Caffe layer parameter into `[stride_h, stride_w]`.
 *
 * @param param Layer spatial parameter containing stride information.
 * @returns Array of 2 stride values.
 */
function resolveStride(param: CaffeSpatialParam | undefined): number[] {
  if (!param) return [1, 1];
  let sh = 1;
  let sw = 1;
  if (param.stride !== undefined) {
    const s = Array.isArray(param.stride) ? (param.stride[0] ?? 1) : param.stride;
    sh = s;
    sw = s;
  } else {
    if (param.stride_h !== undefined) sh = param.stride_h;
    if (param.stride_w !== undefined) sw = param.stride_w;
  }
  return [sh, sw];
}

/**
 * Extracts shape dimensions from a `CaffeBlob` or numeric placeholder.
 * @param blob Blob entry from `CaffeLayerParam.blobs`.
 * @returns Shape dimension array.
 */
function extractBlobShape(blob: CaffeBlob | number | undefined): number[] {
  if (blob !== undefined && typeof blob !== 'number' && blob.shape) {
    if (Array.isArray(blob.shape)) {
      return blob.shape;
    }
    if (Array.isArray(blob.shape.dim)) {
      return blob.shape.dim;
    }
  }
  return [1];
}

/**
 * Maps parsed Caffe layers (`CaffeLayerParam`) to ONNX `Node` instances.
 */
export class CaffeMapper {
  /**
   * Maps a single Caffe layer to an array of ONNX nodes using the registered mapping function.
   *
   * @param layer Caffe layer parameter to map.
   * @param graph Target ONNX Graph.
   * @returns Array of generated ONNX nodes.
   */
  map(layer: CaffeLayerParam, graph: Graph): Node[] {
    const type = layer.type ?? '';
    this.processBlobs(layer, graph);
    const handler = caffeRegistry[type];
    if (handler) {
      return handler(layer, graph);
    }

    const reporter = { warn: console.warn };
    if (type === 'VisionTransform') {
      return [
        new Node(
          'CustomOp',
          layer.bottom || [],
          layer.top || [],
          { domain: new Attribute('domain', 'STRING', 'caffe') },
          layer.name || 'vision_transform',
        ),
      ];
    }

    reporter.warn('Unrecognized Caffe layer');
    return [
      new Node(
        'Identity',
        layer.bottom && layer.bottom[0] !== undefined ? [layer.bottom[0]] : [],
        layer.top && layer.top[0] !== undefined ? [layer.top[0]] : [],
        {},
        layer.name || 'fallback_identity',
      ),
    ];
  }

  /**
   * Maps a Caffe `Convolution` layer to an ONNX `Conv` node.
   *
   * @param layer Parsed Caffe Convolution layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX `Conv` node.
   */
  @register_caffe_op('caffe', 'Convolution')
  mapConvolution(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const param = layer.convolution_param;
    const pads = resolvePadding(param);
    const kernel_shape = resolveKernel(param);
    const strides = resolveStride(param);
    let dilations = [1, 1];
    if (param?.dilation !== undefined) {
      const d = Array.isArray(param.dilation) ? (param.dilation[0] ?? 1) : param.dilation;
      dilations = [d, d];
    }
    const group = typeof param?.group === 'number' ? param.group : 1;

    const layerName = layer.name ?? '';
    const inputs = [...(layer.bottom || [])];
    if (layer.blobs && layer.blobs.length > 0) {
      const wName = `${layerName}_W`;
      inputs.push(wName);
      if (layer.blobs.length > 1) {
        const bName = `${layerName}_B`;
        inputs.push(bName);
      }
    }

    const node = new Node(
      'Conv',
      inputs,
      layer.top || [layerName],
      {
        pads: new Attribute('pads', 'INTS', pads),
        kernel_shape: new Attribute('kernel_shape', 'INTS', kernel_shape),
        strides: new Attribute('strides', 'INTS', strides),
        dilations: new Attribute('dilations', 'INTS', dilations),
        group: new Attribute('group', 'INT', group),
      },
      layerName,
    );
    return [node];
  }

  /**
   * Maps a Caffe `InnerProduct` layer to an ONNX `Gemm` node.
   *
   * @param layer Parsed Caffe InnerProduct layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX `Gemm` node.
   */
  @register_caffe_op('caffe', 'InnerProduct')
  mapInnerProduct(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const inputs = [...(layer.bottom || [])];
    if (layer.blobs && layer.blobs.length > 0) {
      const wName = `${layerName}_W`;
      inputs.push(wName);
      if (layer.blobs.length > 1) {
        const bName = `${layerName}_B`;
        inputs.push(bName);
      }
    }
    const node = new Node(
      'Gemm',
      inputs,
      layer.top || [layerName],
      {
        alpha: new Attribute('alpha', 'FLOAT', 1.0),
        beta: new Attribute('beta', 'FLOAT', 1.0),
        transB: new Attribute('transB', 'INT', 1),
      },
      layerName,
    );
    return [node];
  }

  /**
   * Maps a Caffe `ReLU` layer to an ONNX `Relu` or `LeakyRelu` node.
   *
   * @param layer Parsed Caffe ReLU layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX `Relu` or `LeakyRelu` node.
   */
  @register_caffe_op('caffe', 'ReLU')
  mapReLU(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const param = layer.relu_param || {};
    const negative_slope = param.negative_slope || 0;
    if (negative_slope !== 0) {
      const node = new Node(
        'LeakyRelu',
        layer.bottom || [],
        layer.top || [layerName],
        {
          alpha: new Attribute('alpha', 'FLOAT', negative_slope),
        },
        layerName,
      );
      return [node];
    }
    const node = new Node('Relu', layer.bottom || [], layer.top || [layerName], {}, layerName);
    return [node];
  }

  /**
   * Maps a Caffe `Pooling` layer to an ONNX `MaxPool` or `AveragePool` node.
   *
   * @param layer Parsed Caffe Pooling layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX Pool node.
   */
  @register_caffe_op('caffe', 'Pooling')
  mapPooling(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const param = layer.pooling_param;
    const pads = resolvePadding(param);
    const kernel_shape = resolveKernel(param);
    const strides = resolveStride(param);

    const poolType = param?.pool === 1 ? 'AveragePool' : 'MaxPool';

    const node = new Node(
      poolType,
      layer.bottom || [],
      layer.top || [layerName],
      {
        pads: new Attribute('pads', 'INTS', pads),
        kernel_shape: new Attribute('kernel_shape', 'INTS', kernel_shape),
        strides: new Attribute('strides', 'INTS', strides),
      },
      layerName,
    );
    return [node];
  }

  /**
   * Maps a Caffe `LRN` layer to an ONNX `LRN` node.
   *
   * @param layer Parsed Caffe LRN layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX `LRN` node.
   */
  @register_caffe_op('caffe', 'LRN')
  mapLRN(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const param = layer.lrn_param || {};
    const size = param.local_size !== undefined ? param.local_size : 5;
    const alpha = param.alpha !== undefined ? param.alpha : 1.0;
    const beta = param.beta !== undefined ? param.beta : 0.75;
    const k = param.k !== undefined ? param.k : 1.0;

    const node = new Node(
      'LRN',
      layer.bottom || [],
      layer.top || [layerName],
      {
        size: new Attribute('size', 'INT', size),
        alpha: new Attribute('alpha', 'FLOAT', alpha),
        beta: new Attribute('beta', 'FLOAT', beta),
        bias: new Attribute('bias', 'FLOAT', k),
      },
      layerName,
    );
    return [node];
  }

  /**
   * Maps a Caffe `Softmax` layer to an ONNX `Softmax` node.
   *
   * @param layer Parsed Caffe Softmax layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX `Softmax` node.
   */
  @register_caffe_op('caffe', 'Softmax')
  mapSoftmax(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const param = layer.softmax_param || {};
    const axis = param.axis !== undefined ? param.axis : 1;
    const node = new Node(
      'Softmax',
      layer.bottom || [],
      layer.top || [layerName],
      {
        axis: new Attribute('axis', 'INT', axis),
      },
      layerName,
    );
    return [node];
  }

  /**
   * Maps a Caffe `Eltwise` layer to an ONNX `Add`, `Mul`, or `Max` node.
   *
   * @param layer Parsed Caffe Eltwise layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX node.
   */
  @register_caffe_op('caffe', 'Eltwise')
  mapEltwise(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const param = layer.eltwise_param || {};
    const op = param.operation !== undefined ? param.operation : 1;
    let opType = 'Add';
    if (op === 0) opType = 'Mul';
    else if (op === 2) opType = 'Max';

    const node = new Node(opType, layer.bottom || [], layer.top || [layerName], {}, layerName);
    return [node];
  }

  /**
   * Maps a Caffe `Concat` layer to an ONNX `Concat` node.
   *
   * @param layer Parsed Caffe Concat layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX `Concat` node.
   */
  @register_caffe_op('caffe', 'Concat')
  mapConcat(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const param = layer.concat_param || {};
    const axis = param.axis !== undefined ? param.axis : 1;
    const node = new Node(
      'Concat',
      layer.bottom || [],
      layer.top || [layerName],
      {
        axis: new Attribute('axis', 'INT', axis),
      },
      layerName,
    );
    return [node];
  }

  /**
   * Maps a Caffe `Scale` layer to an ONNX `Mul` (and optional `Add`) node.
   *
   * @param layer Parsed Caffe Scale layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the generated ONNX nodes.
   */
  @register_caffe_op('caffe', 'Scale')
  mapScale(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const inputs = [...(layer.bottom || [])];
    if (layer.blobs && layer.blobs.length > 0) {
      inputs.push(`${layerName}_scale`);
    } else {
      inputs.push(`${layerName}_scale_in`);
    }

    const nodes: Node[] = [];
    const mulTop =
      layer.blobs && layer.blobs.length > 1 ? `${layerName}_mul` : layer.top?.[0] || layerName;
    nodes.push(new Node('Mul', inputs, [mulTop], {}, `${layerName}_Mul`));

    if (layer.blobs && layer.blobs.length > 1) {
      nodes.push(
        new Node(
          'Add',
          [mulTop, `${layerName}_B`],
          layer.top || [layerName],
          {},
          `${layerName}_Add`,
        ),
      );
    }
    return nodes;
  }

  /**
   * Maps a Caffe `BatchNorm` layer to an ONNX `BatchNormalization` node.
   *
   * @param layer Parsed Caffe BatchNorm layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX `BatchNormalization` node.
   */
  @register_caffe_op('caffe', 'BatchNorm')
  mapBatchNorm(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const param = layer.batch_norm_param || {};
    const eps = param.eps !== undefined ? param.eps : 1e-5;
    const inputs = [...(layer.bottom || [])];

    inputs.push(`${layerName}_scale`, `${layerName}_B`, `${layerName}_mean`, `${layerName}_var`);

    const node = new Node(
      'BatchNormalization',
      inputs,
      layer.top || [layerName],
      {
        epsilon: new Attribute('epsilon', 'FLOAT', eps),
      },
      layerName,
    );
    return [node];
  }

  /**
   * Maps a Caffe `Dropout` layer to an ONNX `Dropout` node.
   *
   * @param layer Parsed Caffe Dropout layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX `Dropout` node.
   */
  @register_caffe_op('caffe', 'Dropout')
  mapDropout(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const param = layer.dropout_param || {};
    const ratio = param.dropout_ratio !== undefined ? param.dropout_ratio : 0.5;
    const node = new Node(
      'Dropout',
      layer.bottom || [],
      layer.top || [layerName],
      {
        ratio: new Attribute('ratio', 'FLOAT', ratio),
      },
      layerName,
    );
    return [node];
  }

  /**
   * Maps a Caffe `Reshape` layer to an ONNX `Reshape` node.
   *
   * @param layer Parsed Caffe Reshape layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX `Reshape` node.
   */
  @register_caffe_op('caffe', 'Reshape')
  mapReshape(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const inputs = [...(layer.bottom || [])];
    inputs.push(`${layerName}_shape`);
    const node = new Node('Reshape', inputs, layer.top || [layerName], {}, layerName);
    return [node];
  }

  /**
   * Maps a Caffe `Flatten` layer to an ONNX `Flatten` node.
   *
   * @param layer Parsed Caffe Flatten layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX `Flatten` node.
   */
  @register_caffe_op('caffe', 'Flatten')
  mapFlatten(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const param = layer.flatten_param || {};
    const axis = param.axis !== undefined ? param.axis : 1;
    const node = new Node(
      'Flatten',
      layer.bottom || [],
      layer.top || [layerName],
      {
        axis: new Attribute('axis', 'INT', axis),
      },
      layerName,
    );
    return [node];
  }

  /**
   * Maps a Caffe `Split` layer to ONNX `Identity` node(s).
   *
   * @param layer Parsed Caffe Split layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX `Identity` nodes.
   */
  @register_caffe_op('caffe', 'Split')
  mapSplit(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const nodes: Node[] = [];
    const tops = layer.top || [];
    const bottoms = layer.bottom || [];
    if (tops.length > 0) {
      for (const [i, topName] of tops.entries()) {
        nodes.push(new Node('Identity', bottoms, [topName], {}, `${layerName}_${i}`));
      }
    } else {
      nodes.push(new Node('Identity', bottoms, [layerName], {}, layerName));
    }
    return nodes;
  }

  /**
   * Maps a Caffe `Slice` layer to an ONNX `Split` node.
   *
   * @param layer Parsed Caffe Slice layer.
   * @param _graph Target ONNX Graph.
   * @returns Array containing the corresponding ONNX `Split` node.
   */
  @register_caffe_op('caffe', 'Slice')
  mapSlice(layer: CaffeLayerParam, _graph: Graph): Node[] {
    const layerName = layer.name ?? '';
    const param = layer.slice_param || {};
    const axis = param.axis !== undefined ? param.axis : 1;
    const splitAttr: number[] = [];
    if (param.slice_point !== undefined) {
      const points = Array.isArray(param.slice_point) ? param.slice_point : [param.slice_point];
      splitAttr.push(...points);
    }
    const attrs: Record<string, Attribute> = {
      axis: new Attribute('axis', 'INT', axis),
    };
    if (splitAttr.length > 0) {
      attrs.split = new Attribute('split', 'INTS', splitAttr);
    }
    const node = new Node('Split', layer.bottom || [], layer.top || [], attrs, layerName);
    return [node];
  }

  /**
   * Extracts weight and bias blobs from a Caffe layer and registers them as initializers in the graph.
   * @param layer Caffe layer parameter containing optional `blobs`.
   * @param graph Target ONNX graph.
   */
  private processBlobs(layer: CaffeLayerParam, graph: Graph): void {
    if (!layer.blobs || layer.blobs.length === 0) return;

    const layerName = layer.name ?? '';
    const wName = `${layerName}_W`;
    const wBlob = layer.blobs[0] as CaffeBlob | number;
    const wShape = extractBlobShape(wBlob);
    const wTensor = new Tensor(wName, wShape, 'float32', true, false);
    const wSize = wShape.reduce((a: number, b: number) => a * Math.abs(b), 1) || 1;
    const wBuf = new Float32Array(wSize);
    if (typeof wBlob !== 'number' && Array.isArray(wBlob.data)) {
      wBuf.set(wBlob.data.slice(0, wSize));
    }
    wTensor.data = wBuf;

    graph.initializers.push(wName);
    graph.tensors[wName] = wTensor;

    if (layer.blobs.length > 1) {
      const bName = `${layerName}_B`;
      const bBlob = layer.blobs[1] as CaffeBlob | number;
      const bShape = extractBlobShape(bBlob);
      const bTensor = new Tensor(bName, bShape, 'float32', true, false);
      const bSize = bShape.reduce((a: number, b: number) => a * Math.abs(b), 1) || 1;
      const bBuf = new Float32Array(bSize);
      if (typeof bBlob !== 'number' && Array.isArray(bBlob.data)) {
        bBuf.set(bBlob.data.slice(0, bSize));
      }
      bTensor.data = bBuf;

      graph.initializers.push(bName);
      graph.tensors[bName] = bTensor;
    }
  }
}
