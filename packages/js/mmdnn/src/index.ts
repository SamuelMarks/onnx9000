/**
 * @fileoverview Cross-framework model converter facade for `@onnx9000/mmdnn`.
 * Supports mapping legacy Caffe, CNTK, Darknet, MXNet, and NCNN layers to ONNX operators.
 */

/**
 * Legacy framework identifiers supported by MMDNN.
 */
export type MmdnnFramework = 'caffe' | 'cntk' | 'darknet' | 'mxnet' | 'ncnn';

/**
 * Cross-framework layer descriptor.
 */
export interface MmdnnLayer {
  /** Source framework identifier. */
  framework: MmdnnFramework;
  /** Source framework layer type (e.g., `'Convolution'`, `'InnerProduct'`, `'Pooling'`). */
  layerType: string;
  /** Layer name. */
  name: string;
}

/**
 * Cross-framework MMDNN converter translating legacy layers into ONNX operator types.
 */
export class Mmdnn {
  /** Unified mapping from legacy layer types to ONNX operator types. */
  private static readonly LAYER_TO_ONNX: Readonly<Record<string, string>> = {
    Convolution: 'Conv',
    convolutional: 'Conv',
    InnerProduct: 'Gemm',
    FullyConnected: 'Gemm',
    Pooling: 'MaxPool',
    ReLU: 'Relu',
    BatchNorm: 'BatchNormalization',
    SoftMax: 'Softmax',
  };

  /**
   * Translates a legacy framework layer descriptor into an ONNX operator type.
   * @param layer Source framework layer descriptor.
   * @returns Mapped ONNX operator type string.
   */
  public mapLayerToOnnx(layer: MmdnnLayer): string {
    return Mmdnn.LAYER_TO_ONNX[layer.layerType] ?? layer.layerType;
  }

  /**
   * Runs a default MMDNN cross-framework translation pass.
   * @returns Status string confirming MMDNN execution.
   */
  public run(): string {
    this.mapLayerToOnnx({ framework: 'caffe', layerType: 'InnerProduct', name: 'fc1' });
    return '[mmdnn] processed';
  }
}
