/* eslint-disable */
// @ts-nocheck
import { Graph, Node, Attribute, Tensor } from '@onnx9000/core';

/**
 * A type representing a function that maps a Caffe layer to one or more ONNX Nodes.
 *
 * @param {object} layer - The parsed Caffe layer object.
 * @param {Graph} graph - The target ONNX Graph to which the nodes belong.
 * @returns {Node[]} An array of ONNX nodes representing the converted layer.
 */
type MapperFn = (layer: object, graph: Graph) => Node[];

/**
 * A registry storing mapping functions for supported Caffe layer types.
 *
 * @type {Record<string, MapperFn>}
 */
const caffeRegistry: Record<string, MapperFn> = {};

/**
 * A decorator that registers a method as the mapping function for a specific Caffe layer type.
 *
 * @param {string} domain - The framework domain (e.g., 'caffe').
 * @param {string} opType - The Caffe layer type to register the mapping for (e.g., 'Convolution').
 * @returns {(target: object, propertyKey: string, descriptor: PropertyDescriptor) => void} The decorator function.
 */
export function register_caffe_op(domain: string, opType: string) {
  return function (target: object, propertyKey: string, descriptor: PropertyDescriptor) {
    if (domain === 'caffe') {
      caffeRegistry[opType] = descriptor.value.bind(target);
    }
  };
}

/**
 * Resolves padding values from a Caffe layer parameter into a [pad_top, pad_left, pad_bottom, pad_right] format.
 *
 * @param {object} param - The layer parameter containing padding information.
 * @returns {number[]} An array of 4 padding values.
 */
function resolvePadding(param: object): number[] {
  if (!param) return [0, 0, 0, 0];
  let pad_h = 0,
    pad_w = 0;
  if (param.pad !== undefined) {
    const p = Array.isArray(param.pad) ? param.pad[0] : param.pad;
    pad_h = p;
    pad_w = p;
  } else {
    if (param.pad_h !== undefined) pad_h = param.pad_h;
    if (param.pad_w !== undefined) pad_w = param.pad_w;
  }
  return [pad_h, pad_w, pad_h, pad_w];
}

/**
 * Resolves kernel size values from a Caffe layer parameter into a [kernel_h, kernel_w] format.
 *
 * @param {object} param - The layer parameter containing kernel size information.
 * @returns {number[]} An array of 2 kernel size values.
 */
function resolveKernel(param: object): number[] {
  if (!param) return [1, 1];
  let kh = 1,
    kw = 1;
  if (param.kernel_size !== undefined) {
    const k = Array.isArray(param.kernel_size) ? param.kernel_size[0] : param.kernel_size;
    kh = k;
    kw = k;
  } else {
    if (param.kernel_h !== undefined) kh = param.kernel_h;
    if (param.kernel_w !== undefined) kw = param.kernel_w;
  }
  return [kh, kw];
}

/**
 * Resolves stride values from a Caffe layer parameter into a [stride_h, stride_w] format.
 *
 * @param {object} param - The layer parameter containing stride information.
 * @returns {number[]} An array of 2 stride values.
 */
function resolveStride(param: object): number[] {
  if (!param) return [1, 1];
  let sh = 1,
    sw = 1;
  if (param.stride !== undefined) {
    const s = Array.isArray(param.stride) ? param.stride[0] : param.stride;
    sh = s;
    sw = s;
  } else {
    if (param.stride_h !== undefined) sh = param.stride_h;
    if (param.stride_w !== undefined) sw = param.stride_w;
  }
  return [sh, sw];
}

/**
 * A class responsible for mapping parsed Caffe layers to ONNX nodes.
 */
export class CaffeMapper {
  /**
   * Maps a single Caffe layer to an array of ONNX nodes using the registered mapping function.
   *
   * @param {object} layer - The Caffe layer object to map.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array of generated ONNX nodes, or an empty array if the layer type is unsupported.
   */
  map(layer: object, graph: Graph): Node[] {
    const type = layer.type;
    this.processBlobs(layer, graph);
    if (caffeRegistry[type]) {
      return caffeRegistry[type](layer, graph);
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

    reporter.warn(`Unrecognized Caffe layer`);
    return [
      new Node(
        'Identity',
        layer.bottom ? [layer.bottom[0]] : [],
        layer.top ? [layer.top[0]] : [],
        {},
        layer.name || 'fallback_identity',
      ),
    ];
  }

  /**
   * Maps a Caffe Convolution layer to an ONNX Conv node.
   *
   * @param {object} layer - The parsed Caffe Convolution layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX Conv node.
   */
  @register_caffe_op('caffe', 'Convolution')
  mapConvolution(layer: object, graph: Graph): Node[] {
    const param = layer.convolution_param || {};
    const pads = resolvePadding(param);
    const kernel_shape = resolveKernel(param);
    const strides = resolveStride(param);
    let dilations = [1, 1];
    if (param.dilation !== undefined) {
      const d = Array.isArray(param.dilation) ? param.dilation[0] : param.dilation;
      dilations = [d, d];
    }
    const group = param.group !== undefined ? param.group : 1;

    // In ONNX, Conv inputs are X, W, [B]
    const inputs = [...(layer.bottom || [])];
    if (layer.blobs && layer.blobs.length > 0) {
      const wName = `${layer.name}_W`;
      inputs.push(wName);
      if (layer.blobs.length > 1) {
        const bName = `${layer.name}_B`;
        inputs.push(bName);
      }
    }

    const node = new Node(
      'Conv',
      inputs,
      layer.top || [layer.name],
      {
        pads: new Attribute('pads', 'INTS', pads),
        kernel_shape: new Attribute('kernel_shape', 'INTS', kernel_shape),
        strides: new Attribute('strides', 'INTS', strides),
        dilations: new Attribute('dilations', 'INTS', dilations),
        group: new Attribute('group', 'INT', group),
      },
      layer.name,
    );
    return [node];
  }

  /**
   * Maps a Caffe InnerProduct layer to an ONNX Gemm node.
   *
   * @param {object} layer - The parsed Caffe InnerProduct layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX Gemm node.
   */
  @register_caffe_op('caffe', 'InnerProduct')
  mapInnerProduct(layer: object, graph: Graph): Node[] {
    const param = layer.inner_product_param || {};
    const inputs = [...(layer.bottom || [])];
    if (layer.blobs && layer.blobs.length > 0) {
      const wName = `${layer.name}_W`;
      inputs.push(wName);
      if (layer.blobs.length > 1) {
        const bName = `${layer.name}_B`;
        inputs.push(bName);
      }
    }
    // Caffe InnerProduct does X * W^T
    const node = new Node(
      'Gemm',
      inputs,
      layer.top || [layer.name],
      {
        alpha: new Attribute('alpha', 'FLOAT', 1.0),
        beta: new Attribute('beta', 'FLOAT', 1.0),
        transB: new Attribute('transB', 'INT', 1),
      },
      layer.name,
    );
    return [node];
  }

  /**
   * Maps a Caffe ReLU layer to an ONNX Relu or LeakyRelu node.
   *
   * @param {object} layer - The parsed Caffe ReLU layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX Relu or LeakyRelu node.
   */
  @register_caffe_op('caffe', 'ReLU')
  mapReLU(layer: object, graph: Graph): Node[] {
    const param = layer.relu_param || {};
    const negative_slope = param.negative_slope || 0;
    if (negative_slope !== 0) {
      const node = new Node(
        'LeakyRelu',
        layer.bottom || [],
        layer.top || [layer.name],
        {
          alpha: new Attribute('alpha', 'FLOAT', negative_slope),
        },
        layer.name,
      );
      return [node];
    } else {
      const node = new Node('Relu', layer.bottom || [], layer.top || [layer.name], {}, layer.name);
      return [node];
    }
  }

  /**
   * Maps a Caffe Pooling layer to an ONNX MaxPool or AveragePool node.
   *
   * @param {object} layer - The parsed Caffe Pooling layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX Pool node.
   */
  @register_caffe_op('caffe', 'Pooling')
  mapPooling(layer: object, graph: Graph): Node[] {
    const param = layer.pooling_param || {};
    const pads = resolvePadding(param);
    const kernel_shape = resolveKernel(param);
    const strides = resolveStride(param);

    // pool = 0 (MAX), 1 (AVE), 2 (STOCHASTIC)
    const poolType = param.pool === 1 ? 'AveragePool' : 'MaxPool';

    const node = new Node(
      poolType,
      layer.bottom || [],
      layer.top || [layer.name],
      {
        pads: new Attribute('pads', 'INTS', pads),
        kernel_shape: new Attribute('kernel_shape', 'INTS', kernel_shape),
        strides: new Attribute('strides', 'INTS', strides),
      },
      layer.name,
    );
    return [node];
  }

  /**
   * Maps a Caffe LRN layer to an ONNX LRN node.
   *
   * @param {object} layer - The parsed Caffe LRN layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX LRN node.
   */
  @register_caffe_op('caffe', 'LRN')
  mapLRN(layer: object, graph: Graph): Node[] {
    const param = layer.lrn_param || {};
    const size = param.local_size !== undefined ? param.local_size : 5;
    const alpha = param.alpha !== undefined ? param.alpha : 1.0;
    const beta = param.beta !== undefined ? param.beta : 0.75;
    const k = param.k !== undefined ? param.k : 1.0;

    const node = new Node(
      'LRN',
      layer.bottom || [],
      layer.top || [layer.name],
      {
        size: new Attribute('size', 'INT', size),
        alpha: new Attribute('alpha', 'FLOAT', alpha),
        beta: new Attribute('beta', 'FLOAT', beta),
        bias: new Attribute('bias', 'FLOAT', k),
      },
      layer.name,
    );
    return [node];
  }

  /**
   * Maps a Caffe Softmax layer to an ONNX Softmax node.
   *
   * @param {object} layer - The parsed Caffe Softmax layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX Softmax node.
   */
  @register_caffe_op('caffe', 'Softmax')
  mapSoftmax(layer: object, graph: Graph): Node[] {
    const param = layer.softmax_param || {};
    const axis = param.axis !== undefined ? param.axis : 1;
    const node = new Node(
      'Softmax',
      layer.bottom || [],
      layer.top || [layer.name],
      {
        axis: new Attribute('axis', 'INT', axis),
      },
      layer.name,
    );
    return [node];
  }

  /**
   * Maps a Caffe Eltwise layer to an ONNX Add, Mul, or Max node.
   *
   * @param {object} layer - The parsed Caffe Eltwise layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX node.
   */
  @register_caffe_op('caffe', 'Eltwise')
  mapEltwise(layer: object, graph: Graph): Node[] {
    const param = layer.eltwise_param || {};
    // operation: 0 (PROD), 1 (SUM), 2 (MAX)
    const op = param.operation !== undefined ? param.operation : 1;
    let opType = 'Add';
    if (op === 0) opType = 'Mul';
    else if (op === 2) opType = 'Max';

    const node = new Node(opType, layer.bottom || [], layer.top || [layer.name], {}, layer.name);
    return [node];
  }

  /**
   * Maps a Caffe Concat layer to an ONNX Concat node.
   *
   * @param {object} layer - The parsed Caffe Concat layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX Concat node.
   */
  @register_caffe_op('caffe', 'Concat')
  mapConcat(layer: object, graph: Graph): Node[] {
    const param = layer.concat_param || {};
    const axis = param.axis !== undefined ? param.axis : 1;
    const node = new Node(
      'Concat',
      layer.bottom || [],
      layer.top || [layer.name],
      {
        axis: new Attribute('axis', 'INT', axis),
      },
      layer.name,
    );
    return [node];
  }

  /**
   * Maps a Caffe Scale layer to an ONNX Mul (and optional Add) node.
   *
   * @param {object} layer - The parsed Caffe Scale layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the generated ONNX nodes.
   */
  @register_caffe_op('caffe', 'Scale')
  mapScale(layer: object, graph: Graph): Node[] {
    const inputs = [...(layer.bottom || [])];
    if (layer.blobs && layer.blobs.length > 0) {
      inputs.push(`${layer.name}_scale`);
    } else {
      inputs.push(`${layer.name}_scale_in`);
    }

    const nodes: Node[] = [];
    const mulTop =
      layer.blobs && layer.blobs.length > 1 ? `${layer.name}_mul` : layer.top?.[0] || layer.name;
    nodes.push(new Node('Mul', inputs, [mulTop], {}, `${layer.name}_Mul`));

    if (layer.blobs && layer.blobs.length > 1) {
      nodes.push(
        new Node(
          'Add',
          [mulTop, `${layer.name}_B`],
          layer.top || [layer.name],
          {},
          `${layer.name}_Add`,
        ),
      );
    }
    return nodes;
  }

  /**
   * Maps a Caffe BatchNorm layer to an ONNX BatchNormalization node.
   *
   * @param {object} layer - The parsed Caffe BatchNorm layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX BatchNormalization node.
   */
  @register_caffe_op('caffe', 'BatchNorm')
  mapBatchNorm(layer: object, graph: Graph): Node[] {
    const param = layer.batch_norm_param || {};
    const eps = param.eps !== undefined ? param.eps : 1e-5;
    const inputs = [...(layer.bottom || [])];

    // ONNX needs X, scale, B, mean, var
    inputs.push(
      `${layer.name}_scale`,
      `${layer.name}_B`,
      `${layer.name}_mean`,
      `${layer.name}_var`,
    );

    const node = new Node(
      'BatchNormalization',
      inputs,
      layer.top || [layer.name],
      {
        epsilon: new Attribute('epsilon', 'FLOAT', eps),
      },
      layer.name,
    );
    return [node];
  }

  /**
   * Maps a Caffe Dropout layer to an ONNX Dropout node.
   *
   * @param {object} layer - The parsed Caffe Dropout layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX Dropout node.
   */
  @register_caffe_op('caffe', 'Dropout')
  mapDropout(layer: object, graph: Graph): Node[] {
    const param = layer.dropout_param || {};
    const ratio = param.dropout_ratio !== undefined ? param.dropout_ratio : 0.5;
    // In many ONNX conversions, dropout is often mapped to Identity if ratio=0 or for inference
    // We map it to Dropout as requested
    const node = new Node(
      'Dropout',
      layer.bottom || [],
      layer.top || [layer.name],
      {
        ratio: new Attribute('ratio', 'FLOAT', ratio),
      },
      layer.name,
    );
    return [node];
  }

  /**
   * Maps a Caffe Reshape layer to an ONNX Reshape node.
   *
   * @param {object} layer - The parsed Caffe Reshape layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX Reshape node.
   */
  @register_caffe_op('caffe', 'Reshape')
  mapReshape(layer: object, graph: Graph): Node[] {
    const inputs = [...(layer.bottom || [])];
    inputs.push(`${layer.name}_shape`);
    const node = new Node('Reshape', inputs, layer.top || [layer.name], {}, layer.name);
    return [node];
  }

  /**
   * Maps a Caffe Flatten layer to an ONNX Flatten node.
   *
   * @param {object} layer - The parsed Caffe Flatten layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX Flatten node.
   */
  @register_caffe_op('caffe', 'Flatten')
  mapFlatten(layer: object, graph: Graph): Node[] {
    const param = layer.flatten_param || {};
    const axis = param.axis !== undefined ? param.axis : 1;
    const node = new Node(
      'Flatten',
      layer.bottom || [],
      layer.top || [layer.name],
      {
        axis: new Attribute('axis', 'INT', axis),
      },
      layer.name,
    );
    return [node];
  }

  /**
   * Maps a Caffe Split layer to ONNX Identity node(s).
   *
   * @param {object} layer - The parsed Caffe Split layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX Identity nodes.
   */
  @register_caffe_op('caffe', 'Split')
  mapSplit(layer: object, graph: Graph): Node[] {
    // Caffe Split acts as a pass-through copying data to multiple tops
    const nodes: Node[] = [];
    const tops = layer.top || [];
    const bottoms = layer.bottom || [];
    if (tops.length > 0) {
      for (let i = 0; i < tops.length; i++) {
        nodes.push(new Node('Identity', bottoms, [tops[i]], {}, `${layer.name}_${i}`));
      }
    } else {
      nodes.push(new Node('Identity', bottoms, [layer.name], {}, layer.name));
    }
    return nodes;
  }

  /**
   * Maps a Caffe Slice layer to an ONNX Split node.
   *
   * @param {object} layer - The parsed Caffe Slice layer.
   * @param {Graph} graph - The target ONNX Graph.
   * @returns {Node[]} An array containing the corresponding ONNX Split node.
   */
  @register_caffe_op('caffe', 'Slice')
  mapSlice(layer: object, graph: Graph): Node[] {
    const param = layer.slice_param || {};
    const axis = param.axis !== undefined ? param.axis : 1;
    // Caffe slice points determine the cut sizes. In ONNX this maps to Split.
    const splitAttr: number[] = [];
    if (param.slice_point) {
      const points = Array.isArray(param.slice_point) ? param.slice_point : [param.slice_point];
      splitAttr.push(...points); // Placeholder for length vs point conversion
    }
    const attrs: Record<string, Attribute> = {
      axis: new Attribute('axis', 'INT', axis),
    };
    if (splitAttr.length > 0) {
      attrs.split = new Attribute('split', 'INTS', splitAttr);
    }
    const node = new Node('Split', layer.bottom || [], layer.top || [], attrs, layer.name);
    return [node];
  }

  private processBlobs(layer: object, graph: Graph) {
    if (!layer.blobs || layer.blobs.length === 0) return;

    const wName = `${layer.name}_W`;
    if (layer.blobs.length > 0) {
      const wBlob = layer.blobs[0];
      const shape = wBlob.shape?.dim || [1]; // dummy shape if missing
      const tensor = new Tensor(wName, shape, 'float32', true, false);
      const size = shape.reduce((a: number, b: number) => a * Math.abs(b), 1) || 1;
      tensor.data = new Float32Array(size); // Zero initialized
      if (wBlob.data) {
        (tensor.data as Float32Array).set(wBlob.data.slice(0, size));
      }
      graph.initializers.push(wName);
      graph.tensors[wName] = tensor;
    }
    if (layer.blobs.length > 1) {
      const bName = `${layer.name}_B`;
      const bBlob = layer.blobs[1];
      const shape = bBlob.shape?.dim || [1];
      const tensor = new Tensor(bName, shape, 'float32', true, false);
      const size = shape.reduce((a: number, b: number) => a * Math.abs(b), 1) || 1;
      tensor.data = new Float32Array(size);
      if (bBlob.data) {
        (tensor.data as Float32Array).set(bBlob.data.slice(0, size));
      }
      graph.initializers.push(bName);
      graph.tensors[bName] = tensor;
    }
  }
}
