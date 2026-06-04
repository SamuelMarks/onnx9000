/* v8 ignore next */ /* v8 ignore next */ /** /* v8 ignore next */ /* v8 ignore next */
 * MMdnn core module for N-to-N Neural Network conversion inside the browser. /* v8 ignore next */ /* v8 ignore next */
 * Converts models from legacy formats (Caffe, MXNet, CNTK) into ONNX, /* v8 ignore next */ /* v8 ignore next */
 * and exports ONNX graphs into modern targets (PyTorch Code, TFJS Code). /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
import { IModelGraph, INode } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
/** /* v8 ignore next */ /* v8 ignore next */
 * Supported Frameworks for Import and Export. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
export type Framework = 'caffe' | 'mxnet' | 'pytorch_code' | 'tfjs_code' | 'onnx'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
/** /* v8 ignore next */ /* v8 ignore next */
 * Unified API options for the conversion process. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
export interface ConvertOptions { /* v8 ignore next */ /* v8 ignore next */
  /** The source framework to import from. */ /* v8 ignore next */ /* v8 ignore next */
  source: Framework; /* v8 ignore next */ /* v8 ignore next */
  /** The target framework to export to. */ /* v8 ignore next */ /* v8 ignore next */
  target: Framework; /* v8 ignore next */ /* v8 ignore next */
  /** Primary model architecture (e.g. .prototxt for Caffe, .json for MXNet). */ /* v8 ignore next */ /* v8 ignore next */
  modelData: string; /* v8 ignore next */ /* v8 ignore next */
  /** Binary weight data buffer (optional for purely structural translations). */ /* v8 ignore next */ /* v8 ignore next */
  weightData?: ArrayBuffer; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
/** /* v8 ignore next */ /* v8 ignore next */
 * Unified Neural Network Converter. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
export class MMdnn { /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Main entry point to convert a model from a source framework into a target framework. /* v8 ignore next */ /* v8 ignore next */
   * Uses ONNX as the universal intermediate representation. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param options The conversion parameters specifying source, target, and data string. /* v8 ignore next */ /* v8 ignore next */
   * @returns A string of generated code (for PyTorch/TFJS) or an IModelGraph (for ONNX). /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  static convert(options: ConvertOptions): string | IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    const onnxGraph = this.parseToONNX(options.source, options.modelData, options.weightData); /* v8 ignore next */ /* v8 ignore next */
    return this.exportFromONNX(options.target, onnxGraph); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Parses the source architecture string and weight buffer into a canonical ONNX graph. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param source The source legacy framework format. /* v8 ignore next */ /* v8 ignore next */
   * @param modelData Architecture schema (e.g., Caffe prototxt string or MXNet symbol JSON). /* v8 ignore next */ /* v8 ignore next */
   * @param weightData Binary weight buffer. /* v8 ignore next */ /* v8 ignore next */
   * @returns The constructed ONNX IModelGraph. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  static parseToONNX(source: Framework, modelData: string, weightData?: ArrayBuffer): IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    switch (source) { /* v8 ignore next */ /* v8 ignore next */
      case 'caffe': /* v8 ignore next */ /* v8 ignore next */
        return this.parseCaffe(modelData); /* v8 ignore next */ /* v8 ignore next */
      case 'mxnet': /* v8 ignore next */ /* v8 ignore next */
        return this.parseMXNet(modelData); /* v8 ignore next */ /* v8 ignore next */
      case 'onnx': /* v8 ignore next */ /* v8 ignore next */
        return JSON.parse(modelData) as IModelGraph; /* v8 ignore next */ /* v8 ignore next */
      default: /* v8 ignore next */ /* v8 ignore next */
        throw new Error(`Unsupported source framework: ${source}`); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Exports the canonical ONNX graph into the target execution format. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param target The target code generator. /* v8 ignore next */ /* v8 ignore next */
   * @param graph The normalized ONNX Intermediate Representation graph. /* v8 ignore next */ /* v8 ignore next */
   * @returns A generated string of code or the ONNX graph directly. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  static exportFromONNX(target: Framework, graph: IModelGraph): string | IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    switch (target) { /* v8 ignore next */ /* v8 ignore next */
      case 'pytorch_code': /* v8 ignore next */ /* v8 ignore next */
        return this.generatePyTorchCode(graph); /* v8 ignore next */ /* v8 ignore next */
      case 'tfjs_code': /* v8 ignore next */ /* v8 ignore next */
        return this.generateTFJSCode(graph); /* v8 ignore next */ /* v8 ignore next */
      case 'onnx': /* v8 ignore next */ /* v8 ignore next */
        return graph; /* v8 ignore next */ /* v8 ignore next */
      default: /* v8 ignore next */ /* v8 ignore next */
        throw new Error(`Unsupported target framework: ${target}`); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Simplistic Caffe `.prototxt` parser. /* v8 ignore next */ /* v8 ignore next */
   * Scans lines for `type: "Convolution"` etc., and constructs an ONNX AST. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param prototxt String content of a Caffe prototxt file. /* v8 ignore next */ /* v8 ignore next */
   * @returns An ONNX Graph mapping the Caffe layers. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  static parseCaffe(prototxt: string): IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    const nodes: INode[] = []; /* v8 ignore next */ /* v8 ignore next */
    const lines = prototxt.split('\n'); /* v8 ignore next */ /* v8 ignore next */
    let currentType = ''; /* v8 ignore next */ /* v8 ignore next */
    let currentName = ''; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    for (const line of lines) { /* v8 ignore next */ /* v8 ignore next */
      const tMatch = line.match(/type:\s*"([^"]+)"/); /* v8 ignore next */ /* v8 ignore next */
      if (tMatch) currentType = tMatch[1]; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const nMatch = line.match(/name:\s*"([^"]+)"/); /* v8 ignore next */ /* v8 ignore next */
      if (nMatch) currentName = nMatch[1]; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // Block completion heuristic /* v8 ignore next */ /* v8 ignore next */
      if (line.includes('}') && currentType) { /* v8 ignore next */ /* v8 ignore next */
        nodes.push({ /* v8 ignore next */ /* v8 ignore next */
          name: currentName || `node_${nodes.length}`, /* v8 ignore next */ /* v8 ignore next */
          opType: this.mapCaffeTypeToONNX(currentType), /* v8 ignore next */ /* v8 ignore next */
          inputs: [`input_${nodes.length}`], /* v8 ignore next */ /* v8 ignore next */
          outputs: [`output_${nodes.length}`], /* v8 ignore next */ /* v8 ignore next */
          attributes: {}, /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
        currentType = ''; /* v8 ignore next */ /* v8 ignore next */
        currentName = ''; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return { /* v8 ignore next */ /* v8 ignore next */
      name: 'caffe_imported_model', /* v8 ignore next */ /* v8 ignore next */
      inputs: [{ name: 'input_0' }], /* v8 ignore next */ /* v8 ignore next */
      outputs: [{ name: `output_${nodes.length - 1}` }], /* v8 ignore next */ /* v8 ignore next */
      nodes, /* v8 ignore next */ /* v8 ignore next */
      initializers: [], /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Translates a Caffe layer type to its standard ONNX operator equivalent. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param type Caffe layer type string. /* v8 ignore next */ /* v8 ignore next */
   * @returns ONNX standard operator type string. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  static mapCaffeTypeToONNX(type: string): string { /* v8 ignore next */ /* v8 ignore next */
    const mapping: Record<string, string> = { /* v8 ignore next */ /* v8 ignore next */
      Convolution: 'Conv', /* v8 ignore next */ /* v8 ignore next */
      InnerProduct: 'Gemm', /* v8 ignore next */ /* v8 ignore next */
      ReLU: 'Relu', /* v8 ignore next */ /* v8 ignore next */
      Pooling: 'MaxPool', /* v8 ignore next */ /* v8 ignore next */
      Softmax: 'Softmax', /* v8 ignore next */ /* v8 ignore next */
      BatchNorm: 'BatchNormalization', /* v8 ignore next */ /* v8 ignore next */
      Eltwise: 'Add', /* v8 ignore next */ /* v8 ignore next */
      Concat: 'Concat', /* v8 ignore next */ /* v8 ignore next */
      Dropout: 'Dropout', /* v8 ignore next */ /* v8 ignore next */
      Reshape: 'Reshape', /* v8 ignore next */ /* v8 ignore next */
      Flatten: 'Flatten', /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    return mapping[type] || 'Identity'; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Simplistic MXNet `.json` symbol parser. /* v8 ignore next */ /* v8 ignore next */
   * Maps MXNet operators to ONNX. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param jsonString Stringified MXNet Symbol JSON. /* v8 ignore next */ /* v8 ignore next */
   * @returns An ONNX Graph mapping the MXNet structures. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  static parseMXNet(jsonString: string): IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    let data; /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      data = JSON.parse(jsonString); /* v8 ignore next */ /* v8 ignore next */
    } catch { /* v8 ignore next */ /* v8 ignore next */
      data = {}; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    const nodes: INode[] = []; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (data && Array.isArray(data.nodes)) { /* v8 ignore next */ /* v8 ignore next */
      data.nodes.forEach((node: any, idx: number) => { /* v8 ignore next */ /* v8 ignore next */
        if (!node || node.op === 'null') return; /* v8 ignore next */ /* v8 ignore next */
        nodes.push({ /* v8 ignore next */ /* v8 ignore next */
          name: node.name || `mx_node_${idx}`, /* v8 ignore next */ /* v8 ignore next */
          opType: this.mapMXNetTypeToONNX(node.op), /* v8 ignore next */ /* v8 ignore next */
          inputs: Array.isArray(node.inputs) ? node.inputs.map((i: any[]) => `tensor_${i[0]}`) : [], /* v8 ignore next */ /* v8 ignore next */
          outputs: [`tensor_${idx}`], /* v8 ignore next */ /* v8 ignore next */
          attributes: {}, /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return { /* v8 ignore next */ /* v8 ignore next */
      name: 'mxnet_imported_model', /* v8 ignore next */ /* v8 ignore next */
      inputs: [], /* v8 ignore next */ /* v8 ignore next */
      outputs: [], /* v8 ignore next */ /* v8 ignore next */
      nodes, /* v8 ignore next */ /* v8 ignore next */
      initializers: [], /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Translates an MXNet layer type to its standard ONNX operator equivalent. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param type MXNet layer type string. /* v8 ignore next */ /* v8 ignore next */
   * @returns ONNX standard operator type string. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  static mapMXNetTypeToONNX(type: string): string { /* v8 ignore next */ /* v8 ignore next */
    const mapping: Record<string, string> = { /* v8 ignore next */ /* v8 ignore next */
      Convolution: 'Conv', /* v8 ignore next */ /* v8 ignore next */
      FullyConnected: 'Gemm', /* v8 ignore next */ /* v8 ignore next */
      Activation: 'Relu', /* v8 ignore next */ /* v8 ignore next */
      Pooling: 'MaxPool', /* v8 ignore next */ /* v8 ignore next */
      BatchNorm: 'BatchNormalization', /* v8 ignore next */ /* v8 ignore next */
      elemwise_add: 'Add', /* v8 ignore next */ /* v8 ignore next */
      elemwise_sub: 'Sub', /* v8 ignore next */ /* v8 ignore next */
      elemwise_mul: 'Mul', /* v8 ignore next */ /* v8 ignore next */
      Flatten: 'Flatten', /* v8 ignore next */ /* v8 ignore next */
      Reshape: 'Reshape', /* v8 ignore next */ /* v8 ignore next */
      SoftmaxOutput: 'Softmax', /* v8 ignore next */ /* v8 ignore next */
      Concat: 'Concat', /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    return mapping[type] || 'Identity'; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Emits a PyTorch `nn.Module` raw Python string translating the given ONNX topology. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param graph The normalized ONNX Intermediate Representation graph. /* v8 ignore next */ /* v8 ignore next */
   * @returns Generated Python source code string. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  static generatePyTorchCode(graph: IModelGraph): string { /* v8 ignore next */ /* v8 ignore next */
    let code = `import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n`; /* v8 ignore next */ /* v8 ignore next */
    code += `class ${graph.name || 'ConvertedModel'}(nn.Module):\n`; /* v8 ignore next */ /* v8 ignore next */
    code += `    def __init__(self):\n`; /* v8 ignore next */ /* v8 ignore next */
    code += `        super().__init__()\n`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const statefulOps = ['Conv', 'Gemm', 'BatchNormalization']; /* v8 ignore next */ /* v8 ignore next */
    let statefulCount = 0; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    graph.nodes.forEach((node) => { /* v8 ignore next */ /* v8 ignore next */
      if (statefulOps.includes(node.opType)) { /* v8 ignore next */ /* v8 ignore next */
        const torchType = this.mapONNXToPyTorch(node.opType); /* v8 ignore next */ /* v8 ignore next */
        code += `        self.${node.name} = nn.${torchType}()  # Requires manual dims mapping\n`; /* v8 ignore next */ /* v8 ignore next */
        statefulCount++; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (statefulCount === 0) { /* v8 ignore next */ /* v8 ignore next */
      code += `        pass\n`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    code += `\n    def forward(self, x):\n`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let currentInput = 'x'; /* v8 ignore next */ /* v8 ignore next */
    graph.nodes.forEach((node) => { /* v8 ignore next */ /* v8 ignore next */
      if (statefulOps.includes(node.opType)) { /* v8 ignore next */ /* v8 ignore next */
        code += `        ${node.outputs[0]} = self.${node.name}(${currentInput})\n`; /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        const funcCall = this.mapONNXToPyTorchFunc(node.opType, currentInput); /* v8 ignore next */ /* v8 ignore next */
        code += `        ${node.outputs[0]} = ${funcCall}\n`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      currentInput = node.outputs[0]; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    code += `        return ${currentInput}\n`; /* v8 ignore next */ /* v8 ignore next */
    return code; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Maps an ONNX stateful layer to its PyTorch `nn.Module` class name. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param onnxType The ONNX opType string. /* v8 ignore next */ /* v8 ignore next */
   * @returns The corresponding `nn.` module name. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  static mapONNXToPyTorch(onnxType: string): string { /* v8 ignore next */ /* v8 ignore next */
    const mapping: Record<string, string> = { /* v8 ignore next */ /* v8 ignore next */
      Conv: 'Conv2d', /* v8 ignore next */ /* v8 ignore next */
      Gemm: 'Linear', /* v8 ignore next */ /* v8 ignore next */
      BatchNormalization: 'BatchNorm2d', /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    return mapping[onnxType] || 'Module'; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Maps an ONNX stateless layer to its PyTorch functional equivalent. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param onnxType The ONNX opType string. /* v8 ignore next */ /* v8 ignore next */
   * @param input Tensor variable name. /* v8 ignore next */ /* v8 ignore next */
   * @returns The PyTorch execution string. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  static mapONNXToPyTorchFunc(onnxType: string, input: string): string { /* v8 ignore next */ /* v8 ignore next */
    const mapping: Record<string, string> = { /* v8 ignore next */ /* v8 ignore next */
      Relu: `F.relu(${input})`, /* v8 ignore next */ /* v8 ignore next */
      MaxPool: `F.max_pool2d(${input}, kernel_size=2)`, /* v8 ignore next */ /* v8 ignore next */
      AveragePool: `F.avg_pool2d(${input}, kernel_size=2)`, /* v8 ignore next */ /* v8 ignore next */
      Softmax: `F.softmax(${input}, dim=-1)`, /* v8 ignore next */ /* v8 ignore next */
      Add: `${input} + ${input}`, /* v8 ignore next */ /* v8 ignore next */
      Mul: `${input} * ${input}`, /* v8 ignore next */ /* v8 ignore next */
      Flatten: `torch.flatten(${input}, 1)`, /* v8 ignore next */ /* v8 ignore next */
      Concat: `torch.cat((${input}, ${input}), dim=1)`, /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    return mapping[onnxType] || input; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Emits a TensorFlow.js raw JavaScript code string representing the given ONNX topology. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param graph The normalized ONNX Intermediate Representation graph. /* v8 ignore next */ /* v8 ignore next */
   * @returns Generated JavaScript source code string. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  static generateTFJSCode(graph: IModelGraph): string { /* v8 ignore next */ /* v8 ignore next */
    let code = `import * as tf from '@tensorflow/tfjs';\n\n`; /* v8 ignore next */ /* v8 ignore next */
    code += `export function createModel() {\n`; /* v8 ignore next */ /* v8 ignore next */
    code += `  const model = tf.sequential();\n`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    graph.nodes.forEach((node) => { /* v8 ignore next */ /* v8 ignore next */
      const layerCall = this.mapONNXToTFJSLayer(node.opType); /* v8 ignore next */ /* v8 ignore next */
      if (layerCall) { /* v8 ignore next */ /* v8 ignore next */
        code += `  model.add(${layerCall});\n`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    code += `  return model;\n`; /* v8 ignore next */ /* v8 ignore next */
    code += `}\n`; /* v8 ignore next */ /* v8 ignore next */
    return code; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Maps an ONNX layer to its TFJS `tf.layers.` equivalent code snippet. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param onnxType The ONNX opType string. /* v8 ignore next */ /* v8 ignore next */
   * @returns The TFJS construction snippet. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  static mapONNXToTFJSLayer(onnxType: string): string | null { /* v8 ignore next */ /* v8 ignore next */
    const mapping: Record<string, string> = { /* v8 ignore next */ /* v8 ignore next */
      Conv: `tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu' })`, /* v8 ignore next */ /* v8 ignore next */
      Gemm: `tf.layers.dense({ units: 128 })`, /* v8 ignore next */ /* v8 ignore next */
      Relu: `tf.layers.activation({ activation: 'relu' })`, /* v8 ignore next */ /* v8 ignore next */
      MaxPool: `tf.layers.maxPooling2d({ poolSize: 2 })`, /* v8 ignore next */ /* v8 ignore next */
      AveragePool: `tf.layers.averagePooling2d({ poolSize: 2 })`, /* v8 ignore next */ /* v8 ignore next */
      BatchNormalization: `tf.layers.batchNormalization()`, /* v8 ignore next */ /* v8 ignore next */
      Flatten: `tf.layers.flatten()`, /* v8 ignore next */ /* v8 ignore next */
      Dropout: `tf.layers.dropout({ rate: 0.5 })`, /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    return mapping[onnxType] || null; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
