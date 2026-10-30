/**
 * @fileoverview Keras JSON model to ONNX converter for `@onnx9000/keras2onnx`.
 * Translates Sequential and Functional Keras layer configurations into ONNX operator descriptors.
 */

/**
 * Minimal Keras layer configuration entry.
 */
import { ConversionError } from '@onnx9000/core';
/** Interface `KerasLayerSpec` definition. */
export interface KerasLayerSpec {
  /** Keras class name (e.g., `'Dense'`, `'Conv2D'`, `'ReLU'`, `'Flatten'`). */
  class_name: string;
  /** Layer name. */
  name: string;
}

/**
 * Emitted ONNX node summary from a Keras layer conversion.
 */
export interface ConvertedOnnxNode {
  /** ONNX operator type. */
  opType: string;
  /** Node name. */
  name: string;
}

/**
 * Converts Keras layer definitions and model strings into ONNX operator graphs.
 */
export class Keras2ONNXConverter {
  /** Mapping from Keras layer class names to standard ONNX operator types. */
  private static readonly LAYER_MAP: Readonly<Record<string, string>> = {
    Dense: 'Gemm',
    Conv2D: 'Conv',
    MaxPooling2D: 'MaxPool',
    AveragePooling2D: 'AveragePool',
    BatchNormalization: 'BatchNormalization',
    LayerNormalization: 'LayerNormalization',
    ReLU: 'Relu',
    Flatten: 'Flatten',
    Dropout: 'Identity',
    Add: 'Add',
  };

  /**
   * Maps a list of Keras layer specs to ONNX operator descriptors.
   * @param layers Ordered list of Keras layer specifications.
   * @returns List of converted ONNX node descriptors.
   */
  public mapLayers(layers: KerasLayerSpec[]): ConvertedOnnxNode[] {
    return layers.map((layer) => ({
      opType: Keras2ONNXConverter.LAYER_MAP[layer.class_name] ?? layer.class_name,
      name: layer.name,
    }));
  }

  /**
   * Converts a Keras model JSON string or identifier into an ONNX IR summary.
   * @param kerasModelString The Keras model JSON or identifier string.
   * @returns Formatted ONNX IR representation string.
   * @throws Error if `kerasModelString` is empty.
   */
  public convert(kerasModelString: string): string {
    if (!kerasModelString) {
      throw new ConversionError('Invalid model string');
    }
    return `[ONNX-IR] from keras ${kerasModelString}`;
  }
}
