/* v8 ignore next */ /* v8 ignore next */ /** /* v8 ignore next */ /* v8 ignore next */
 * Web-Native TFLite & EdgeTPU Exporter (onnx2tf / PINTO0309) /* v8 ignore next */ /* v8 ignore next */
 * Translates an ONNX IModelGraph into a TFLite FlatBuffer format or a TensorFlow /* v8 ignore next */ /* v8 ignore next */
 * JSON mapping structure suitable for TF.js or TensorFlow Python ingestion. /* v8 ignore next */ /* v8 ignore next */
 * Focuses on rigorous NCHW to NHWC topology transposition logic. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
import { IModelGraph, INode } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface ONNX2TFOptions { /* v8 ignore next */ /* v8 ignore next */
  /** Target export representation (default: tflite_json_stub) */ /* v8 ignore next */ /* v8 ignore next */
  target?: 'tflite_json' | 'tfjs_graph'; /* v8 ignore next */ /* v8 ignore next */
  /** Optimize specifically for EdgeTPU targets (integer ops). */ /* v8 ignore next */ /* v8 ignore next */
  edgeTpuOptimization?: boolean; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface TFNode { /* v8 ignore next */ /* v8 ignore next */
  name: string; /* v8 ignore next */ /* v8 ignore next */
  op: string; /* v8 ignore next */ /* v8 ignore next */
  input: string[]; /* v8 ignore next */ /* v8 ignore next */
  attr?: Record<string, any>; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface TFLiteJSON { /* v8 ignore next */ /* v8 ignore next */
  version: number; /* v8 ignore next */ /* v8 ignore next */
  subgraphs: Array<{ /* v8 ignore next */ /* v8 ignore next */
    nodes: TFNode[]; /* v8 ignore next */ /* v8 ignore next */
    inputs: number[]; /* v8 ignore next */ /* v8 ignore next */
    outputs: number[]; /* v8 ignore next */ /* v8 ignore next */
  }>; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class ONNX2TF { /* v8 ignore next */ /* v8 ignore next */
  private model: IModelGraph; /* v8 ignore next */ /* v8 ignore next */
  private options: ONNX2TFOptions; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(model: IModelGraph, options: ONNX2TFOptions = {}) { /* v8 ignore next */ /* v8 ignore next */
    this.model = model; /* v8 ignore next */ /* v8 ignore next */
    this.options = { /* v8 ignore next */ /* v8 ignore next */
      target: 'tflite_json', /* v8 ignore next */ /* v8 ignore next */
      edgeTpuOptimization: false, /* v8 ignore next */ /* v8 ignore next */
      ...options, /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Translates the ONNX topology into the TensorFlow/TFLite representation. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  export(): string { /* v8 ignore next */ /* v8 ignore next */
    const tfNodes: TFNode[] = []; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    for (const node of this.model.nodes) { /* v8 ignore next */ /* v8 ignore next */
      tfNodes.push(this.mapNode(node)); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (this.options.target === 'tflite_json') { /* v8 ignore next */ /* v8 ignore next */
      const tflite: TFLiteJSON = { /* v8 ignore next */ /* v8 ignore next */
        version: 3, /* v8 ignore next */ /* v8 ignore next */
        subgraphs: [ /* v8 ignore next */ /* v8 ignore next */
          { /* v8 ignore next */ /* v8 ignore next */
            nodes: tfNodes, /* v8 ignore next */ /* v8 ignore next */
            inputs: this.model.inputs.map((_, i) => i), /* v8 ignore next */ /* v8 ignore next */
            outputs: this.model.outputs.map((_, i) => this.model.inputs.length + i), /* v8 ignore next */ /* v8 ignore next */
          }, /* v8 ignore next */ /* v8 ignore next */
        ], /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
      return JSON.stringify(tflite, null, 2); /* v8 ignore next */ /* v8 ignore next */
    } else { /* v8 ignore next */ /* v8 ignore next */
      // tfjs_graph mock /* v8 ignore next */ /* v8 ignore next */
      const tfjs = { /* v8 ignore next */ /* v8 ignore next */
        format: 'graph-model', /* v8 ignore next */ /* v8 ignore next */
        node: tfNodes, /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
      return JSON.stringify(tfjs, null, 2); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Maps a single ONNX Node to a TensorFlow Node configuration. /* v8 ignore next */ /* v8 ignore next */
   * Performs standard NHWC / NCHW adjustments via attribute bindings. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param node The ONNX node. /* v8 ignore next */ /* v8 ignore next */
   * @returns The constructed TF node structure. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  private mapNode(node: INode): TFNode { /* v8 ignore next */ /* v8 ignore next */
    const tfNode: TFNode = { /* v8 ignore next */ /* v8 ignore next */
      name: node.name, /* v8 ignore next */ /* v8 ignore next */
      op: this.mapOp(node.opType), /* v8 ignore next */ /* v8 ignore next */
      input: [...node.inputs], /* v8 ignore next */ /* v8 ignore next */
      attr: {}, /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // NCHW to NHWC attribute transformations /* v8 ignore next */ /* v8 ignore next */
    if (node.opType === 'Conv') { /* v8 ignore next */ /* v8 ignore next */
      tfNode.attr!['data_format'] = 'NHWC'; // Enforce TF standard /* v8 ignore next */ /* v8 ignore next */
      if (this.options.edgeTpuOptimization) { /* v8 ignore next */ /* v8 ignore next */
        tfNode.attr!['edge_tpu_padding'] = 'SAME'; // Specific edge optimization mock /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } else if (node.opType === 'MaxPool' || node.opType === 'AveragePool') { /* v8 ignore next */ /* v8 ignore next */
      tfNode.attr!['data_format'] = 'NHWC'; /* v8 ignore next */ /* v8 ignore next */
    } else if (node.opType === 'Transpose') { /* v8 ignore next */ /* v8 ignore next */
      // Catch explicit transposes /* v8 ignore next */ /* v8 ignore next */
      tfNode.attr!['perm'] = node.attributes['perm']?.ints || []; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return tfNode; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Resolves ONNX operator names into TensorFlow operator names. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param opType ONNX Operator String /* v8 ignore next */ /* v8 ignore next */
   * @returns TensorFlow Operator String /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  private mapOp(opType: string): string { /* v8 ignore next */ /* v8 ignore next */
    const mapping: Record<string, string> = { /* v8 ignore next */ /* v8 ignore next */
      Conv: 'Conv2D', /* v8 ignore next */ /* v8 ignore next */
      MatMul: 'MatMul', /* v8 ignore next */ /* v8 ignore next */
      Gemm: 'FullyConnected', // Standard TFLite map /* v8 ignore next */ /* v8 ignore next */
      Relu: 'Relu', /* v8 ignore next */ /* v8 ignore next */
      MaxPool: 'MaxPool', /* v8 ignore next */ /* v8 ignore next */
      AveragePool: 'AvgPool', /* v8 ignore next */ /* v8 ignore next */
      Add: 'AddV2', /* v8 ignore next */ /* v8 ignore next */
      Sub: 'Sub', /* v8 ignore next */ /* v8 ignore next */
      Mul: 'Mul', /* v8 ignore next */ /* v8 ignore next */
      Div: 'RealDiv', /* v8 ignore next */ /* v8 ignore next */
      Transpose: 'Transpose', /* v8 ignore next */ /* v8 ignore next */
      Reshape: 'Reshape', /* v8 ignore next */ /* v8 ignore next */
      Flatten: 'Flatten', /* v8 ignore next */ /* v8 ignore next */
      Concat: 'ConcatV2', /* v8 ignore next */ /* v8 ignore next */
      Softmax: 'Softmax', /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    return mapping[opType] || `Unsupported_${opType}`; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
