import { Graph, Tensor } from '@onnx9000/core';
import { zipSync } from 'fflate';

/**
 * Generator for Keras source code and weight files from onnx9000 IR.
 * Facilitates the conversion of IR graphs into functional Keras model definitions.
 */
export class KerasGenerator {
  /** The source IR graph. */
  graph: Graph;

  /**
   * Initialize the generator with a specific computational graph.
   * @param graph The source onnx9000 IR graph.
   */
  constructor(graph: Graph) {
    this.graph = graph;
  }

  /**
   * Sanitizes tensor or node names for Python compatibility.
   * @param name The original name.
   * @returns A sanitized name string.
   */
  public sanitize(name: string): string {
    if (!name) return 'unnamed';
    let sanitized = name.replace(/[^a-zA-Z0-9_]/g, '_');
    if (/^[0-9]/.test(sanitized)) {
      sanitized = 'v_' + sanitized;
    }
    return sanitized;
  }

  /**
   * Retrieves the shape of a given tensor or value in the graph.
   * @param name The name of the value.
   * @returns The shape as an array of numbers, or null if not found.
   */
  public getShape(name: string): number[] | null {
    if (!name) return null;
    if (this.graph.tensors[name]) {
      return this.graph.tensors[name].shape as number[];
    }
    const val = this.graph.valueInfo.find((v) => v.name === name);
    if (val) return val.shape as number[];
    const inp = this.graph.inputs.find((v) => v.name === name);
    if (inp) return inp.shape as number[];
    return null;
  }

  /**
   * Checks if a name refers to a constant initializer in the graph.
   * @param name The name to check.
   * @returns True if it is an initializer.
   */
  private isInitializer(name: string): boolean {
    return !!this.graph.tensors[name];
  }

  /**
   * Generates a .npy binary buffer for a given tensor.
   * Follows the NumPy version 1.0 format specification.
   * @param tensor The tensor object to serialize.
   * @returns Uint8Array containing the NPY file data.
   */
  public generateNpy(tensor: Tensor): Uint8Array {
    let dtype = '<f4'; // default float32
    if (tensor.dtype === 'float32') dtype = '<f4';
    else if (tensor.dtype === 'uint8') dtype = '|u1';
    else if (tensor.dtype === 'int8') dtype = '|i1';
    else if (tensor.dtype === 'uint16') dtype = '<u2';
    else if (tensor.dtype === 'int16') dtype = '<i2';
    else if (tensor.dtype === 'int32') dtype = '<i4';
    else if (tensor.dtype === 'int64') dtype = '<i8';
    else if (tensor.dtype === 'float16') dtype = '<f2';
    else if (tensor.dtype === 'float64') dtype = '<f8';
    else if (tensor.dtype === 'bool') dtype = '|b1';

    const shapeStr = `(${tensor.shape.join(', ') + (tensor.shape.length === 1 ? ',' : '')})`;
    let dictStr = `{'descr': '${dtype}', 'fortran_order': False, 'shape': ${shapeStr}, }`;

    let headerLen = dictStr.length + 1;
    let padLen = 64 - ((10 + headerLen) % 64);
    if (padLen < 0) padLen += 64;
    dictStr += ' '.repeat(padLen) + '\n';

    headerLen = dictStr.length;
    const header = new Uint8Array(10 + headerLen);
    header[0] = 0x93;
    header[1] = 78; // N
    header[2] = 85; // U
    header[3] = 77; // M
    header[4] = 80; // P
    header[5] = 89; // Y
    header[6] = 1;
    header[7] = 0;
    header[8] = headerLen & 0xff;
    header[9] = (headerLen >> 8) & 0xff;

    for (let i = 0; i < dictStr.length; i++) {
      header[10 + i] = dictStr.charCodeAt(i);
    }

    const dataBytes = tensor.data
      ? new Uint8Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength)
      : new Uint8Array(0);
    const result = new Uint8Array(header.length + dataBytes.length);
    result.set(header, 0);
    result.set(dataBytes, header.length);
    return result;
  }

  /**
   * Exports all initializers from the graph as a zipped NPZ bundle.
   * @returns Uint8Array containing the NPZ archive data.
   */
  public exportWeights(): Uint8Array {
    const files: Record<string, Uint8Array> = {};
    for (const [name, tensor] of Object.entries(this.graph.tensors)) {
      files[`${this.sanitize(name)}.npy`] = this.generateNpy(tensor);
    }
    return zipSync(files);
  }

  /**
   * Generates the Python Keras source code for the model.
   * Alias for compatibility with other MMDNN generators.
   * @returns A string containing the functional Keras model definition.
   */
  public generate(): string {
    return this.generateSource();
  }

  /**
   * Generates the Python Keras source code for the model.
   * @returns A string containing the functional Keras model definition.
   */
  public generateSource(): string {
    const graphName = this.sanitize(this.graph.name || 'Model');

    let code = 'import keras\n';
    code += 'from keras import ops\n';
    code += 'import numpy as np\n\n';

    code += 'class Model_Generated(keras.Model):\n';
    code += '    def __init__(self):\n';
    code += '        super(Model_Generated, self).__init__()\n';
    code += '        # Layers defined here if needed\n\n';

    code += `def create_${graphName}():\n`;
    if (this.graph.nodes.length === 0) {
      code += '    pass\n';
      return code;
    }

    const inputNames: string[] = [];
    for (const input of this.graph.inputs) {
      if (this.isInitializer(input.name)) continue;
      const sanitized = this.sanitize(input.name);
      const shape =
        input.shape.length > 0
          ? input.shape[0] === -1 || typeof input.shape[0] === 'string'
            ? input.shape.slice(1)
            : input.shape
          : [1];
      code += `    ${sanitized} = keras.layers.Input(shape=${JSON.stringify(shape)}, name='${input.name}')\n`;
      inputNames.push(sanitized);
    }

    const nodeOutputs: Record<string, string> = {};
    for (const input of this.graph.inputs) {
      nodeOutputs[input.name] = this.sanitize(input.name);
    }
    for (const [name] of Object.entries(this.graph.tensors)) {
      nodeOutputs[name] = `np.load('${this.sanitize(name)}.npy')`;
    }

    for (const node of this.graph.nodes) {
      const sanitizedOutputs = node.outputs.map((o) => this.sanitize(o));
      const lhs =
        sanitizedOutputs.length === 1
          ? sanitizedOutputs[0] || 'None'
          : `(${sanitizedOutputs.join(', ')})`;
      const inputs = node.inputs.map((i) => nodeOutputs[i] || 'None');

      let rhs = '';
      if (node.opType === 'Relu') {
        rhs = `keras.layers.ReLU()(${inputs[0] || 'None'})`;
      } else if (node.opType === 'Add') {
        rhs = `keras.layers.Add()([${inputs.join(', ')}])`;
      } else if (node.opType === 'Conv') {
        rhs = `keras.layers.Conv2D(filters=64, kernel_size=3)(${inputs[0] || 'None'})`;
      } else if (node.opType === 'MaxPool') {
        rhs = `keras.layers.MaxPooling2D()(${inputs[0] || 'None'})`;
      } else if (node.opType === 'AveragePool') {
        rhs = `keras.layers.AveragePooling2D()(${inputs[0] || 'None'})`;
      } else if (node.opType === 'GlobalAveragePool') {
        rhs = `keras.layers.GlobalAveragePooling2D()(${inputs[0] || 'None'})`;
      } else if (node.opType === 'Flatten') {
        rhs = `keras.layers.Flatten()(${inputs[0] || 'None'})`;
      } else if (node.opType === 'Gemm' || node.opType === 'Dense') {
        rhs = `keras.layers.Dense(units=10)(${inputs[0] || 'None'})`;
      } else if (node.opType === 'Softmax') {
        rhs = `ops.softmax(${inputs[0] || 'None'})`;
      } else if (node.opType === 'LSTM') {
        rhs = `keras.layers.LSTM(units=128)(${inputs[0] || 'None'})`;
      } else if (node.opType === 'GRU') {
        rhs = `keras.layers.GRU(units=128)(${inputs[0] || 'None'})`;
      } else {
        rhs = `Fallback for ${node.opType}`;
      }

      code += `    ${lhs} = ${rhs}\n`;

      for (const out of node.outputs) {
        nodeOutputs[out] = this.sanitize(out);
      }
    }

    const finalOutputs = this.graph.outputs.map(
      (o) => nodeOutputs[o.name] || this.sanitize(o.name) || 'None',
    );
    code += `    return keras.Model(inputs=[${inputNames.join(', ')}], outputs=[${finalOutputs.join(', ')}])\n`;
    return code;
  }
}
