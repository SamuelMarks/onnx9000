/**
 * @fileoverview High-level Keras-style Layer and LayersModel abstractions for @onnx9000/tfjs-shim.
 */

import { Tensor } from './tensor.js';

/**
 * Interface for layer configuration.
 */
export interface LayerConfig {
  /** The name of the layer. */
  name?: string;
  /** Number of units in the layer. */
  units?: number;
  /** Additional configuration properties. */
  [key: string]: string | number | boolean | undefined | Record<string, unknown>;
}

/**
 * Interface for model configuration.
 */
export interface ModelConfig {
  /** The inputs of the model. */
  inputs: Tensor | Tensor[] | string | string[];
  /** The outputs of the model. */
  outputs: Tensor | Tensor[] | string | string[];
  /** The name of the model. */
  name?: string;
}

/**
 * Represents a layer in a neural network.
 */
export class Layer {
  /** The name of the layer. */
  name: string;

  /**
   * Creates a new Layer.
   * @param config The layer configuration.
   */
  constructor(config: LayerConfig) {
    this.name = config.name || 'layer';
  }

  /** The internal weights of the layer. */
  private _weights: Tensor[] = [];

  /**
   * Returns the weights of the layer.
   * @returns An array of tensors representing the weights.
   */
  getWeights(): Tensor[] {
    return this._weights;
  }

  /**
   * Sets the weights of the layer.
   * @param weights An array of tensors representing the weights.
   */
  setWeights(weights: Tensor[]): void {
    this._weights = weights;
  }
}

/**
 * A LayersModel is a model composed of layers.
 */
export class LayersModel extends Layer {
  /** The layers in the model. */
  layers: Layer[] = [];
  /** The compiled configuration. */
  _compiledConfig?: Record<string, unknown>;

  /**
   * Adds a layer to the model.
   * @param layer The layer to add.
   */
  add(layer: Layer) {
    this.layers.push(layer);
  }

  /**
   * Configures the model for training.
   * @param config The compilation configuration.
   */
  compile(config: Record<string, unknown>) {
    this._compiledConfig = config;
  }

  /**
   * Executes the model with the given input.
   * @param x The input tensor or array of tensors.
   * @returns The output tensor or array of tensors.
   */
  predict(x: Tensor | Tensor[]): Tensor | Tensor[] {
    return x;
  }

  /**
   * Evaluates the model on a given dataset.
   * @param x The input tensor.
   * @param y The target tensor.
   * @returns A tensor representing the evaluation result.
   */
  evaluate(x: Tensor, y: Tensor): Tensor {
    let sum = 0;
    const len = Math.min(x.size, y.size);
    for (let i = 0; i < len; i++) sum += ((x.nData[i] as number) - (y.nData[i] as number)) ** 2;
    return new Tensor([1], 'float32', [sum / len]);
  }
}

/** Functions for creating various neural network layers. */
export const layers = {
  Layer,
  /** Creates a dense layer. */
  dense: (config: LayerConfig) => new Layer(config),
  /** Creates a 2D convolution layer. */
  conv2d: (config: LayerConfig) => new Layer(config),
  /** Creates a 2D max pooling layer. */
  maxPooling2d: (config: LayerConfig) => new Layer(config),
  /** Creates a flatten layer. */
  flatten: (config: LayerConfig) => new Layer(config),
  /** Creates a dropout layer. */
  dropout: (config: LayerConfig) => new Layer(config),
  /** Creates a batch normalization layer. */
  batchNormalization: (config: LayerConfig) => new Layer(config),
  /** Creates a ReLU layer. */
  reLU: (config: LayerConfig) => new Layer(config),
};

/**
 * Creates a sequential model.
 * @param config Optional configuration for the sequential model.
 * @returns A new LayersModel.
 */
export function sequential(config?: LayerConfig): LayersModel {
  return new LayersModel(config || {});
}

/**
 * Creates a model with specified inputs and outputs.
 * @param config The model configuration.
 * @returns A new LayersModel.
 */
export function model(config: ModelConfig): LayersModel {
  return new LayersModel(Object(config) as LayerConfig);
}
