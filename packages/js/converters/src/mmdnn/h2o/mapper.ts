/**
 * @fileoverview mapper module for converters.
 */
/**
 * H2O to ONNX IR mapper.
 */

import { Attribute, Graph, Node, ValueInfo } from '@onnx9000/core';

/** Represents the `H2OMapper` class. */
export class H2OMapper {
  /** Property `modelData` of `H2OMapper`. */
  modelData: Record<string, unknown>;

  /**
   * Initializes a new `H2OMapper` instance.
   * @param modelData Parameter `modelData`.
   */
  constructor(modelData: Record<string, unknown>) {
    this.modelData = modelData;
  }

  /**
   * Executes `map` on `H2OMapper`.
   * @returns Result of `map`.
   */
  map(): Graph {
    const graph = new Graph('H2O_Model');
    graph.opsetImports = { '': 14, 'ai.onnx.ml': 3 };

    graph.inputs.push(new ValueInfo('X', [-1, 10], 'float32'));
    graph.outputs.push(new ValueInfo('Y', [-1, 1], 'float32'));

    const algo = typeof this.modelData.algo === 'string' ? this.modelData.algo : '';
    let opType = 'TreeEnsembleRegressor';
    const attrs: Record<string, Attribute> = {};

    if (algo === 'xgboost') {
      opType = 'TreeEnsembleRegressor';
      attrs.n_targets = new Attribute('n_targets', 'INT', 1);
    } else if (algo === 'deeplearning') {
      opType = 'MatMul';
    } else {
      opType = 'TreeEnsembleRegressor';
      attrs.n_targets = new Attribute('n_targets', 'INT', 1);
      attrs.post_transform = new Attribute('post_transform', 'STRING', 'NONE');
    }

    const domain = opType.startsWith('Tree') ? 'ai.onnx.ml' : '';
    const node = new Node(opType, ['X'], ['Y'], attrs, opType, domain);
    graph.addNode(node);

    return graph;
  }
}
