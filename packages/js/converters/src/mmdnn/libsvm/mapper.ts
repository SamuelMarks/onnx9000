/**
 * @fileoverview mapper module for converters.
 */
/**
 * LibSVM to ONNX IR mapper.
 */

import { Attribute, Graph, Node, ValueInfo } from '@onnx9000/core';
import type { LibSVMModel } from './parser.js';

/** Represents the `LibSVMMapper` class. */
export class LibSVMMapper {
  /** Property `model` of `LibSVMMapper`. */
  model: LibSVMModel;

  /**
   * Initializes a new `LibSVMMapper` instance.
   * @param model Parameter `model`.
   */
  constructor(model: LibSVMModel) {
    this.model = model;
  }

  /**
   * Executes `map` on `LibSVMMapper`.
   * @returns Result of `map`.
   */
  map(): Graph {
    const graph = new Graph('LibSVM_Model');
    graph.opsetImports = { '': 14, 'ai.onnx.ml': 3 };

    graph.inputs.push(new ValueInfo('X', [-1, 10], 'float32'));
    graph.outputs.push(new ValueInfo('Y', [-1, 1], 'float32'));

    const ktype = this.model.kernelType.toUpperCase();
    const opType = this.model.svmType.includes('svr') ? 'SVMRegressor' : 'SVMClassifier';

    const attrs: Record<string, Attribute> = {
      kernel_type: new Attribute('kernel_type', 'STRING', ktype),
      rho: new Attribute('rho', 'FLOATS', [this.model.rho]),
    };

    if (this.model.coefs.length > 0) {
      attrs.coefficients = new Attribute('coefficients', 'FLOATS', this.model.coefs);
    }

    const node = new Node(opType, ['X'], ['Y'], attrs, opType, 'ai.onnx.ml');
    graph.addNode(node);

    return graph;
  }
}
