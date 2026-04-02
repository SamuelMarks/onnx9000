/**
 * Common types for MMDNN parsers and mappers.
 */

/** Caffe Blob Proto structure. */
export interface CaffeBlob {
  data: number[];
  shape?: {
    dim: number[];
  };
}

/** Caffe Layer Parameter structure. */
export interface CaffeLayer {
  name?: string;
  type: string;
  bottom?: string[];
  top?: string[];
  blobs?: CaffeBlob[];
  convolution_param?: {
    pad?: number | number[];
    pad_h?: number;
    pad_w?: number;
    kernel_size?: number | number[];
    kernel_h?: number;
    kernel_w?: number;
    stride?: number | number[];
    stride_h?: number;
    stride_w?: number;
    dilation?: number | number[];
    group?: number;
  };
  inner_product_param?: {
    num_output?: number;
    bias_term?: boolean;
    axis?: number;
    transpose?: boolean;
  };
  relu_param?: {
    negative_slope?: number;
  };
  pooling_param?: {
    pool?: number;
    pad?: number | number[];
    pad_h?: number;
    pad_w?: number;
    kernel_size?: number | number[];
    kernel_h?: number;
    kernel_w?: number;
    stride?: number | number[];
    stride_h?: number;
    stride_w?: number;
    global_pooling?: boolean;
  };
  lrn_param?: {
    local_size?: number;
    alpha?: number;
    beta?: number;
    norm_region?: number;
    k?: number;
  };
  softmax_param?: {
    axis?: number;
  };
  eltwise_param?: {
    operation?: number;
    coeff?: number[];
  };
  concat_param?: {
    axis?: number;
  };
  batch_norm_param?: {
    use_global_stats?: boolean;
    moving_average_fraction?: number;
    eps?: number;
  };
  dropout_param?: {
    dropout_ratio?: number;
  };
  reshape_param?: {
    shape?: {
      dim: number[];
    };
  };
  flatten_param?: {
    axis?: number;
    end_axis?: number;
  };
  slice_param?: {
    axis?: number;
    slice_point?: number | number[];
  };
  [key: string]: string | number | boolean | string[] | number[] | object | object[] | undefined;
}

/** MXNet Node structure. */
export interface MxNetNode {
  op: string;
  name: string;
  attrs?: Record<string, string>;
  inputs: [number, number, number][];
}

/** MXNet Symbol structure. */
export interface MxNetSymbol {
  nodes: MxNetNode[];
  arg_nodes: number[];
  heads: [number, number, number][];
}

/** TensorFlow Node structure. */
export interface TFNode {
  name: string;
  op: string;
  input?: string[];
  attr?: Record<string, TFAttr>;
}

/** TensorFlow Attribute structure. */
export interface TFAttr {
  s?: string;
  i?: number;
  f?: number;
  b?: boolean;
  type?: number;
  shape?: {
    dim: { size: number }[];
  };
  tensor?: {
    dtype: number;
    tensor_shape: {
      dim: { size: number }[];
    };
    tensor_content?: Uint8Array;
  };
}

/** Paddle Op structure. */
export interface PaddleOp {
  type: string;
  inputs: Record<string, string[]>;
  outputs: Record<string, string[]>;
  attrs: Record<
    string,
    { type: number; value: string | number | boolean | number[] | string[] | object }
  >;
}

/** Paddle Block structure. */
export interface PaddleBlock {
  ops: PaddleOp[];
}

/** Paddle Model structure. */
export interface PaddleModel {
  blocks: PaddleBlock[];
}

/**
 * Returns a constant string identifying the MMDNN types module.
 * @returns The string 'MMDNN_TYPES'.
 */
export function getTypesIdentifier(): string {
  return 'MMDNN_TYPES';
}
