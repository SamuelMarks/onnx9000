/* eslint-disable */
// @ts-nocheck
/* eslint-disable */
import { Graph, Node } from '@onnx9000/core';
import { zipSync } from 'fflate';

export class KerasGenerator {
  graph: Graph;

  constructor(graph: Graph) {
    this.graph = graph;
  }

  private sanitize(name: string): string {
    if (!name) return 'unnamed';
    let sanitized = name.replace(/[^a-zA-Z0-9_]/g, '_');
    if (/^[0-9]/.test(sanitized)) {
      sanitized = 'v_' + sanitized;
    }
    return sanitized;
  }

  private getShape(name: string): number[] | null {
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

  private isInitializer(name: string): boolean {
    return !!this.graph.tensors[name];
  }

  generateNpy(tensor: any): Uint8Array {
    let dtype = '<f4'; // float32
    if (tensor.dataType === 1) dtype = '<f4';
    else if (tensor.dataType === 2) dtype = '|u1';
    else if (tensor.dataType === 3) dtype = '|i1';
    else if (tensor.dataType === 4) dtype = '<u2';
    else if (tensor.dataType === 5) dtype = '<i2';
    else if (tensor.dataType === 6) dtype = '<i4';
    else if (tensor.dataType === 7) dtype = '<i8';
    else if (tensor.dataType === 10) dtype = '<f2'; // float16
    else if (tensor.dataType === 11) dtype = '<f8'; // float64
    
    const shapeStr = `(${tensor.shape.join(', ') + (tensor.shape.length === 1 ? ',' : '')})`;
    let dictStr = `{'descr': '${dtype}', 'fortran_order': False, 'shape': ${shapeStr}, }`;
    
    // Padding
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
    header[6] = 1; // major
    header[7] = 0; // minor
    header[8] = headerLen & 0xff;
    header[9] = (headerLen >> 8) & 0xff;
    
    for (let i = 0; i < dictStr.length; i++) {
        header[10 + i] = dictStr.charCodeAt(i);
    }
    
    const data = new Uint8Array(tensor.data.buffer || tensor.data);
    const result = new Uint8Array(header.length + data.length);
    result.set(header, 0);
    result.set(data, header.length);
    return result;
  }

  exportWeights(): Uint8Array {
    const files: Record<string, Uint8Array> = {};
    for (const [name, tensor] of Object.entries(this.graph.tensors)) {
        files[this.sanitize(name) + '.npy'] = this.generateNpy(tensor);
    }
    return zipSync(files);
  }

  generate(): string {
    const lines = [
      'import keras',
      'import numpy as np',
      'from keras import ops',
      'from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Add, Activation, BatchNormalization, Lambda, Permute, SimpleRNN, LSTM, GRU, Dot',
      '',
    ];

    const className = `Model_${this.sanitize(this.graph.name || 'Generated')}`;
    lines.push(`class ${className}(keras.Model):`);
    lines.push(`    def __init__(self, **kwargs):`);
    lines.push(`        super().__init__(**kwargs)`);

    const initLines: string[] = [];
    const callLines: string[] = [];
    const weightAssignLines: string[] = [];

    // Inputs
    const trueInputs = this.graph.inputs.filter((inp) => !this.isInitializer(inp.name));
    const inputNames: string[] = [];
    for (const inp of trueInputs) {
      inputNames.push(this.sanitize(inp.name));
    }

    if (inputNames.length > 1) {
       callLines.push(`        ${inputNames.join(', ')} = inputs`);
    } else if (inputNames.length === 1) {
       callLines.push(`        ${inputNames[0]} = inputs`);
    } else {
       callLines.push(`        pass`);
    }

    let layerCounter = 0;
    const nextLayerName = (prefix: string) => `${prefix}_${layerCounter++}`;

    for (const node of this.graph.nodes) {
      const out = this.sanitize(node.outputs[0] || `out_${node.name || 'node'}`);

      switch (node.opType) {
        case 'Conv': {
          const wShape = this.getShape(node.inputs[1]!);
          const outChannels = wShape ? wShape[0] : 32;
          const kernelSize =
            (node.attributes['kernel_shape']?.value as number[]) ||
            (wShape ? wShape.slice(2) : [3, 3]);
          const strides = (node.attributes['strides']?.value as number[]) || [1, 1];
          const pads = (node.attributes['pads']?.value as number[]) || [0, 0, 0, 0];
          let padding = 'valid';
          if ((pads as number[]).some((p: number) => p > 0)) padding = 'same';

          const inp = this.sanitize(node.inputs[0]!);
          const lName = nextLayerName('conv');
          initLines.push(
            `        self.${lName} = Conv2D(filters=${Number(outChannels)}, kernel_size=${JSON.stringify(kernelSize)}, strides=${JSON.stringify(strides)}, padding='${padding}', data_format='channels_first', use_bias=${node.inputs.length > 2 ? 'True' : 'False'}, name='${this.sanitize(node.name || out)}')`
          );
          callLines.push(`        ${out} = self.${lName}(${inp})`);
          
          if (this.isInitializer(node.inputs[1]!)) {
              weightAssignLines.push(`        self.${lName}.kernel.assign(np.transpose(weights['${this.sanitize(node.inputs[1]!)}'], (2, 3, 1, 0)))`);
          }
          if (node.inputs.length > 2 && this.isInitializer(node.inputs[2]!)) {
              weightAssignLines.push(`        self.${lName}.bias.assign(weights['${this.sanitize(node.inputs[2]!)}'])`);
          }
          break;
        }
        case 'MaxPool':
        case 'AveragePool': {
          const poolType = node.opType === 'MaxPool' ? 'MaxPooling2D' : 'AveragePooling2D';
          const kernelSize = (node.attributes['kernel_shape']?.value as number[]) || [2, 2];
          const strides = (node.attributes['strides']?.value as number[]) || [2, 2];
          const pads = (node.attributes['pads']?.value as number[]) || [0, 0, 0, 0];
          let padding = 'valid';
          if ((pads as number[]).some((p: number) => p > 0)) padding = 'same';

          const inp = this.sanitize(node.inputs[0]!);
          const lName = nextLayerName('pool');
          initLines.push(
            `        self.${lName} = ${poolType}(pool_size=${JSON.stringify(kernelSize)}, strides=${JSON.stringify(strides)}, padding='${padding}', data_format='channels_first', name='${this.sanitize(node.name || out)}')`
          );
          callLines.push(`        ${out} = self.${lName}(${inp})`);
          break;
        }
        case 'GlobalAveragePool': {
          const inp = this.sanitize(node.inputs[0]!);
          const lName = nextLayerName('gap');
          initLines.push(
            `        self.${lName} = GlobalAveragePooling2D(data_format='channels_first', name='${this.sanitize(node.name || out)}')`
          );
          callLines.push(`        ${out} = self.${lName}(${inp})`);
          break;
        }
        case 'Relu': {
          const inp = this.sanitize(node.inputs[0]!);
          callLines.push(`        ${out} = ops.relu(${inp})`);
          break;
        }
        case 'Flatten': {
          const inp = this.sanitize(node.inputs[0]!);
          const lName = nextLayerName('flatten');
          initLines.push(`        self.${lName} = Flatten(data_format='channels_first', name='${this.sanitize(node.name || out)}')`);
          callLines.push(`        ${out} = self.${lName}(${inp})`);
          break;
        }
        case 'Gemm':
        case 'MatMul': {
          const inpA = this.sanitize(node.inputs[0]!);
          const inpB = this.sanitize(node.inputs[1]!);
          const wShape = this.getShape(node.inputs[1]!);
          
          if (!this.isInitializer(node.inputs[1]!)) {
             callLines.push(`        ${out} = ops.matmul(${inpA}, ${inpB})`);
          } else {
            const outFeatures = wShape
              ? node.attributes['transB']?.value
                ? wShape[0]
                : wShape[1]
              : 10;
            const lName = nextLayerName('dense');
            initLines.push(
              `        self.${lName} = Dense(units=${Number(outFeatures)}, name='${this.sanitize(node.name || out)}')`
            );
            callLines.push(`        ${out} = self.${lName}(${inpA})`);
            
            if (node.attributes['transB']?.value) {
                weightAssignLines.push(`        self.${lName}.kernel.assign(np.transpose(weights['${this.sanitize(node.inputs[1]!)}']))`);
            } else {
                weightAssignLines.push(`        self.${lName}.kernel.assign(weights['${this.sanitize(node.inputs[1]!)}'])`);
            }
            if (node.inputs.length > 2 && this.isInitializer(node.inputs[2]!)) {
                weightAssignLines.push(`        self.${lName}.bias.assign(weights['${this.sanitize(node.inputs[2]!)}'])`);
            }
          }
          break;
        }
        case 'Add': {
          const inps = node.inputs.map((i) => this.sanitize(i)).join(', ');
          const lName = nextLayerName('add');
          initLines.push(`        self.${lName} = Add(name='${this.sanitize(node.name || out)}')`);
          callLines.push(`        ${out} = self.${lName}([${inps}])`);
          break;
        }
        case 'Softmax': {
          const inp = this.sanitize(node.inputs[0]!);
          callLines.push(`        ${out} = ops.softmax(${inp})`);
          break;
        }
        case 'BatchNormalization': {
          const inp = this.sanitize(node.inputs[0]!);
          const epsilon = node.attributes['epsilon']?.value || 1e-5;
          const momentum = node.attributes['momentum']?.value || 0.9;
          const lName = nextLayerName('bn');
          initLines.push(
            `        self.${lName} = BatchNormalization(axis=1, epsilon=${epsilon}, momentum=${momentum}, name='${this.sanitize(node.name || out)}')`
          );
          callLines.push(`        ${out} = self.${lName}(${inp})`);
          
          if (this.isInitializer(node.inputs[1]!)) {
              weightAssignLines.push(`        self.${lName}.gamma.assign(weights['${this.sanitize(node.inputs[1]!)}'])`);
          }
          if (node.inputs.length > 2 && this.isInitializer(node.inputs[2]!)) {
              weightAssignLines.push(`        self.${lName}.beta.assign(weights['${this.sanitize(node.inputs[2]!)}'])`);
          }
          if (node.inputs.length > 3 && this.isInitializer(node.inputs[3]!)) {
              weightAssignLines.push(`        self.${lName}.moving_mean.assign(weights['${this.sanitize(node.inputs[3]!)}'])`);
          }
          if (node.inputs.length > 4 && this.isInitializer(node.inputs[4]!)) {
              weightAssignLines.push(`        self.${lName}.moving_variance.assign(weights['${this.sanitize(node.inputs[4]!)}'])`);
          }
          break;
        }
        case 'Shape': {
          const inp = this.sanitize(node.inputs[0]!);
          callLines.push(`        ${out} = ops.shape(${inp})`);
          break;
        }
        case 'Gather': {
          const data = this.sanitize(node.inputs[0]!);
          const indices = this.sanitize(node.inputs[1]!);
          const axis = node.attributes['axis']?.value || 0;
          callLines.push(`        ${out} = ops.take(${data}, ${indices}, axis=${axis})`);
          break;
        }
        case 'Unsqueeze': {
          const data = this.sanitize(node.inputs[0]!);
          const axes = node.attributes['axes']?.value;
          if (axes) {
            callLines.push(`        ${out} = ops.expand_dims(${data}, axis=${(axes as number[])[0]})`);
          } else {
             const axesInp = this.sanitize(node.inputs[1]!);
             callLines.push(`        ${out} = ops.expand_dims(${data}, axis=${axesInp}[0])`);
          }
          break;
        }
        case 'Squeeze': {
          const data = this.sanitize(node.inputs[0]!);
          const axes = node.attributes['axes']?.value;
          if (axes) {
            callLines.push(`        ${out} = ops.squeeze(${data}, axis=${JSON.stringify(axes)})`);
          } else {
             const axesInp = this.sanitize(node.inputs[1]!);
             if (axesInp && axesInp !== 'unnamed') {
                 callLines.push(`        ${out} = ops.squeeze(${data}, axis=${axesInp})`);
             } else {
                 callLines.push(`        ${out} = ops.squeeze(${data})`);
             }
          }
          break;
        }
        case 'RNN':
        case 'LSTM':
        case 'GRU': {
          const inp = this.sanitize(node.inputs[0]!);
          let layerClass = 'SimpleRNN';
          if (node.opType === 'LSTM') layerClass = 'LSTM';
          if (node.opType === 'GRU') layerClass = 'GRU';
          const hiddenSize = node.attributes['hidden_size']?.value || 64;
          const lName = nextLayerName(layerClass.toLowerCase());
          initLines.push(
            `        self.${lName} = ${layerClass}(units=${hiddenSize}, return_sequences=True, return_state=True, name='${this.sanitize(node.name || out)}')`
          );
          callLines.push(`        ${out}_tuple = self.${lName}(${inp})`);
          callLines.push(`        ${out} = ${out}_tuple[0]`);
          break;
        }
        case 'If': {
           const cond = this.sanitize(node.inputs[0]!);
           callLines.push(`        ${out} = ops.cond(${cond}, lambda: ops.convert_to_tensor(1.0), lambda: ops.convert_to_tensor(0.0))`);
           break;
        }
        case 'Loop': {
           const inps = node.inputs.map((i) => this.sanitize(i)).join(', ');
           callLines.push(`        ${out} = ops.identity([${inps}][0])`);
           break;
        }
        default: {
          const inps = node.inputs.map((i) => this.sanitize(i)).filter(i => i !== 'unnamed');
          const inputsStr = inps.length > 0 ? `[${inps.join(', ')}]` : `[]`;
          callLines.push(`        ${out} = ${inputsStr}[0] if len(${inputsStr}) > 0 else ops.convert_to_tensor(0.0)  # Fallback for ${String(node.opType)}`);
        }
      }
    }

    if (initLines.length === 0) {
      initLines.push('        pass');
    }
    for (const l of initLines) {
      lines.push(l);
    }

    lines.push('');
    lines.push(`    def load_weights(self, npz_path):`);
    lines.push(`        weights = np.load(npz_path)`);
    for (const l of weightAssignLines) {
      lines.push(l);
    }
    if (weightAssignLines.length === 0) {
      lines.push('        pass');
    }

    lines.push('');
    lines.push(`    def call(self, inputs):`);
    
    if (callLines.length === 0) {
      callLines.push('        pass');
    }
    for (const l of callLines) {
      lines.push(l);
    }

    const outputs = this.graph.outputs.map((o) => this.sanitize(o.name)).join(', ');
    if (outputs) {
      lines.push(`        return [${outputs}]`);
    } else {
      lines.push(`        return None`);
    }

    lines.push('');
    lines.push(`def create_${this.sanitize(this.graph.name || 'model')}():`);
    lines.push(`    model = ${className}()`);
    lines.push(`    return model`);

    return lines.join('\n') + '\n';
  }
}
