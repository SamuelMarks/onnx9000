/**
 * @fileoverview Common intermediate node and model interfaces for MMDNN parsers and mappers.
 * Defines strongly typed representations for Caffe, MXNet, PaddlePaddle, Darknet, NCNN, and TensorFlow.
 */

/** Caffe Blob Proto structure. */
export interface CaffeBlob {
  /** Flattened floating-point weight or bias values. */
  data: number[];
  /** Shape dimensions of the blob, either as a `BlobShape` object or direct dimension array. */
  shape?: { dim: number[] } | number[];
}

/** Spatial parameter fields shared by Caffe Convolution and Pooling layers. */
export interface CaffeSpatialParam {
  /** Uniform or per-axis padding value(s). */
  pad?: number | number[];
  /** Vertical padding height. */
  pad_h?: number;
  /** Horizontal padding width. */
  pad_w?: number;
  /** Uniform or per-axis kernel size(s). */
  kernel_size?: number | number[];
  /** Vertical kernel height. */
  kernel_h?: number;
  /** Horizontal kernel width. */
  kernel_w?: number;
  /** Uniform or per-axis stride value(s). */
  stride?: number | number[];
  /** Vertical stride height. */
  stride_h?: number;
  /** Horizontal stride width. */
  stride_w?: number;
  /** Dilation factor(s) for convolution. */
  dilation?: number | number[];
  /** Number of convolution groups or boolean flag in legacy prototxt. */
  group?: number | boolean;
  /** Number of output filters. */
  num_output?: number;
  /** Whether a bias term is included. */
  bias_term?: boolean;
  /** Pooling method code (0 = MAX, 1 = AVE, 2 = STOCHASTIC). */
  pool?: number;
  /** Whether global pooling is enabled. */
  global_pooling?: boolean;
}

/** Recursive prototxt value type for parsed Caffe text definitions. */
export type CaffePrototxtValue =
  | string
  | number
  | boolean
  | CaffePrototxtObject
  | CaffePrototxtValue[];

/** Generic key-value object produced when parsing nested Caffe prototxt blocks. */
export interface CaffePrototxtObject {
  [key: string]: CaffePrototxtValue | undefined;
}

/** Caffe Layer Parameter structure (`CaffeLayerParam`). */
export interface CaffeLayerParam {
  /** Layer identifier name. */
  name?: string;
  /** Layer operator type (e.g., `'Convolution'`, `'InnerProduct'`, `'ReLU'`). */
  type?: string;
  /** Input blob names. */
  bottom?: string[];
  /** Output blob names. */
  top?: string[];
  /** Learned parameter blobs (weights and biases) or mock numeric placeholders. */
  blobs?: (CaffeBlob | number)[];
  /** Convolution parameters. */
  convolution_param?: CaffeSpatialParam;
  /** InnerProduct (fully connected) parameters. */
  inner_product_param?: {
    num_output?: number;
    bias_term?: boolean;
    axis?: number;
    transpose?: boolean;
  };
  /** ReLU activation parameters. */
  relu_param?: {
    negative_slope?: number;
  };
  /** Pooling parameters. */
  pooling_param?: CaffeSpatialParam;
  /** Local Response Normalization (LRN) parameters. */
  lrn_param?: {
    local_size?: number;
    alpha?: number;
    beta?: number;
    norm_region?: number;
    k?: number;
  };
  /** Softmax parameters. */
  softmax_param?: {
    axis?: number;
  };
  /** Element-wise operation parameters (0 = PROD, 1 = SUM, 2 = MAX). */
  eltwise_param?: {
    operation?: number;
    coeff?: number[];
  };
  /** Concatenation parameters. */
  concat_param?: {
    axis?: number;
  };
  /** Batch normalization parameters. */
  batch_norm_param?: {
    use_global_stats?: boolean;
    moving_average_fraction?: number;
    eps?: number;
  };
  /** Dropout parameters. */
  dropout_param?: {
    dropout_ratio?: number;
  };
  /** Reshape parameters. */
  reshape_param?: {
    shape?: {
      dim: number[];
    };
  };
  /** Flatten parameters. */
  flatten_param?: {
    axis?: number;
    end_axis?: number;
  };
  /** Slice parameters. */
  slice_param?: {
    axis?: number;
    slice_point?: number | number[];
  };
  /** Phase inclusion rules in prototxt. */
  include?: CaffePrototxtObject[];
  /** Additional arbitrary prototxt fields. */
  [key: string]:
    | CaffePrototxtValue
    | CaffeBlob[]
    | (CaffeBlob | number)[]
    | CaffeSpatialParam
    | undefined;
}

/** Alias for `CaffeLayerParam` for backwards compatibility. */
export type CaffeLayer = CaffeLayerParam;

/** Parsed Caffe NetParameter structure from `.prototxt` or `.caffemodel`. */
export interface CaffeNetParameter {
  /** Network name. */
  name?: string;
  /** Ordered list of layers in the network. */
  layer: CaffeLayerParam[];
  /** Network input blob names. */
  input?: string[];
  /** Legacy network input dimensions. */
  input_dim?: number[];
  /** Network input shape specifications. */
  input_shape?: CaffePrototxtValue[];
  /** Top-level dimension specifications. */
  dim?: CaffePrototxtValue[];
  /** Top-level bottom specifications. */
  bottom?: CaffePrototxtValue[];
  /** Top-level top specifications. */
  top?: CaffePrototxtValue[];
  /** Additional top-level prototxt properties. */
  [key: string]: CaffePrototxtValue | CaffeLayerParam[] | undefined;
}

/** MXNet symbolic graph node structure (`MxNetNode`). */
export interface MxNetNode {
  /** MXNet operator name (e.g., `'Convolution'`, `'null'`, `'FullyConnected'`). */
  op: string;
  /** Node name (optional in partial test fixtures). */
  name?: string;
  /** String-encoded operator attributes. */
  attrs?: Record<string, string>;
  /** Input references as `[node_index, output_index, version]` tuples. */
  inputs?: [number, number, number][];
}

/** MXNet Symbol JSON structure. */
export interface MxNetSymbol {
  /** Nodes in the symbolic execution graph. */
  nodes: MxNetNode[];
  /** Indices of argument (parameter/input) nodes. */
  arg_nodes: number[];
  /** Output head references. */
  heads: [number, number, number][];
}

/** Parsed MXNet NDArray parameter entry from a `.params` binary buffer. */
export interface MxNetParamTensor {
  /** Tensor dimensions. */
  shape: number[];
  /** Typed array view over the tensor values. */
  data: Float32Array | Float64Array | Int32Array;
  /** MXNet numeric dtype code (0 = float32, 1 = float64, 4 = int32). */
  dtype: number;
}

/** TensorFlow Node structure. */
export interface TFNode {
  /** Node name. */
  name: string;
  /** TensorFlow op type. */
  op: string;
  /** Input tensor names. */
  input?: string[];
  /** Node attributes. */
  attr?: Record<string, TFAttr>;
}

/** TensorFlow Attribute structure. */
export interface TFAttr {
  /** String value. */
  s?: string;
  /** Integer value. */
  i?: number;
  /** Float value. */
  f?: number;
  /** Boolean value. */
  b?: boolean;
  /** DataType enum code. */
  type?: number;
  /** Shape attribute value. */
  shape?: {
    dim: { size: number }[];
  };
  /** Tensor attribute value. */
  tensor?: {
    dtype: number;
    tensor_shape: {
      dim: { size: number }[];
    };
    tensor_content?: Uint8Array;
  };
}

/** PaddlePaddle operator attribute dictionary (`PaddleAttrs`). */
export interface PaddleAttrs {
  /** Convolution or pooling padding values. */
  paddings?: number[];
  /** Convolution or pooling stride values. */
  strides?: number[];
  /** Convolution dilation values. */
  dilations?: number[];
  /** Number of convolution groups. */
  groups?: number;
  /** Pooling kernel size. */
  ksize?: number[];
  /** Pooling mode (`'max'` or `'avg'`). */
  pooling_type?: string;
  /** Whether to use ceiling mode in pooling. */
  ceil_mode?: boolean;
  /** Batch normalization epsilon. */
  epsilon?: number;
  /** Batch normalization momentum. */
  momentum?: number;
  /** Number of column dimensions for first matrix in `mul`. */
  x_num_col_dims?: number;
  /** Number of column dimensions for second matrix in `mul`. */
  y_num_col_dims?: number;
  /** Axis for concat or split operations. */
  axis?: number;
  /** Number of sections or section sizes for `split`. */
  num_or_sections?: number | number[];
  /** Whether to transpose input X in `matmul` (lowercase key). */
  transpose_x?: boolean;
  /** Whether to transpose input X in `matmul` (uppercase key). */
  transpose_X?: boolean;
  /** Whether to transpose input Y in `matmul` (lowercase key). */
  transpose_y?: boolean;
  /** Whether to transpose input Y in `matmul` (uppercase key). */
  transpose_Y?: boolean;
  /** Additional operator attributes. */
  [key: string]:
    | string
    | number
    | boolean
    | number[]
    | string[]
    | {
        type: number;
        value: string | number | boolean | number[] | string[];
      }
    | undefined;
}

/** PaddlePaddle operator description (`PaddleOpDesc`). */
export interface PaddleOpDesc {
  /** Operator type name (e.g., `'conv2d'`, `'pool2d'`, `'relu'`). */
  type: string;
  /** Optional operator instance name. */
  name?: string;
  /** Named input parameter slots mapping to variable name arrays. */
  inputs?: Record<string, string[] | undefined>;
  /** Named output parameter slots mapping to variable name arrays. */
  outputs?: Record<string, string[] | undefined>;
  /** Operator attributes. */
  attrs?: PaddleAttrs;
}

/** Alias for `PaddleOpDesc` for backwards compatibility. */
export type PaddleOp = PaddleOpDesc;

/** Paddle Block structure. */
export interface PaddleBlock {
  /** Operators within this program block. */
  ops: PaddleOpDesc[];
}

/** Paddle Model structure. */
export interface PaddleModel {
  /** Optional model name. */
  name?: string;
  /** Program blocks in the Paddle model. */
  blocks?: PaddleBlock[];
  /** Additional top-level metadata fields. */
  [key: string]: string | number | boolean | PaddleBlock[] | undefined;
}

/** Darknet configuration section (`DarknetSection`). */
export interface DarknetSection {
  /** Section/layer type (e.g., `'net'`, `'convolutional'`, `'maxpool'`, `'yolo'`). */
  type: string;
  /** Input/output channel count. */
  channels?: number;
  /** Input height. */
  height?: number;
  /** Input width. */
  width?: number;
  /** Number of convolutional filters. */
  filters?: number;
  /** Connected layer output size. */
  output?: number;
  /** Route layer indices (single index or array of indices). */
  layers?: number | number[];
  /** Kernel size. */
  size?: number;
  /** Stride size. */
  stride?: number;
  /** Padding flag or size. */
  pad?: number;
  /** Whether batch normalization is enabled (`1` = true). */
  batch_normalize?: number;
  /** Activation function name (e.g., `'leaky'`, `'mish'`, `'swish'`, `'linear'`). */
  activation?: string;
  /** Shortcut source layer offset. */
  from?: number;
  /** Additional Darknet section properties parsed from `.cfg`. */
  [key: string]: string | number | (string | number)[] | undefined;
}

/** NCNN layer node representation (`NcnnLayer`). */
export interface NcnnLayer {
  /** NCNN layer type (e.g., `'Input'`, `'Convolution'`, `'Pooling'`). */
  type: string;
  /** Layer instance name. */
  name: string;
  /** Input blob names. */
  bottoms: string[];
  /** Output blob names. */
  tops: string[];
  /** Numeric key-value parameter attributes from `.param`. */
  attrs: Record<string, string>;
}

/**
 * Returns a constant string identifying the MMDNN types module.
 * @returns The string `'MMDNN_TYPES'`.
 */
export function getTypesIdentifier(): string {
  return 'MMDNN_TYPES';
}
