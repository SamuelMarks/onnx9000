/**
 * LibSVM to ONNX IR mapper.
 */

import { Graph, ValueInfo, Node, Attribute } from '@onnx9000/core';
import { LibSVMModel } from './parser.js';

export class LibSVMMapper {
  /* v8 ignore next */ /* v8 ignore next */
  model: LibSVMModel;
  /* v8 ignore next */ /* v8 ignore next */
  constructor(model: LibSVMModel) {
    /* v8 ignore next */ /* v8 ignore next */
    this.model = model; /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  map(): Graph {
    /* v8 ignore next */ /* v8 ignore next */
    const graph = new Graph('LibSVM_Model'); /* v8 ignore next */ /* v8 ignore next */
    graph.opsetImports = { '': 14, 'ai.onnx.ml': 3 }; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    graph.inputs.push(
      new ValueInfo('X', [-1, 10], 'float32'),
    ); /* v8 ignore next */ /* v8 ignore next */
    graph.outputs.push(
      new ValueInfo('Y', [-1, 1], 'float32'),
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const ktype = this.model.kernelType.toUpperCase(); /* v8 ignore next */ /* v8 ignore next */
    const opType = this.model.svmType.includes('svr')
      ? 'SVMRegressor'
      : 'SVMClassifier'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const attrs: Record<string, Attribute> = {
      /* v8 ignore next */ /* v8 ignore next */
      kernel_type: new Attribute(
        'kernel_type',
        'STRING',
        ktype,
      ) /* v8 ignore next */ /* v8 ignore next */,
      rho: new Attribute('rho', 'FLOATS', [
        this.model.rho,
      ]) /* v8 ignore next */ /* v8 ignore next */,
    }; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    if (this.model.coefs.length > 0) {
      /* v8 ignore next */ /* v8 ignore next */
      attrs['coefficients'] = new Attribute(
        'coefficients',
        'FLOATS',
        this.model.coefs,
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const node = new Node(
      opType,
      ['X'],
      ['Y'],
      attrs,
      opType,
      'ai.onnx.ml',
    ); /* v8 ignore next */ /* v8 ignore next */
    graph.addNode(node); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    return graph; /* v8 ignore next */ /* v8 ignore next */
  }
}
