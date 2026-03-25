/* eslint-disable */
import { Graph, Node } from '@onnx9000/core';

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

  generate(): string {
    const lines = [
      'import keras',
      'from keras.models import Model',
      'from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Add, Activation',
      '',
      `def create_${this.sanitize(this.graph.name || 'model')}():`,
    ];

    const initLines: string[] = [];
    const forwardLines: string[] = [];

    // Inputs
    const trueInputs = this.graph.inputs.filter((inp) => !this.isInitializer(inp.name));
    const inputNames: string[] = [];
    for (const inp of trueInputs) {
      const name = this.sanitize(inp.name);
      inputNames.push(name);
      const shape = this.getShape(inp.name) || [224, 224, 3];
      // Keras usually expects shapes without batch dim for Input
      const kerasShape = shape.slice(1);
      forwardLines.push(`${name} = Input(shape=${JSON.stringify(kerasShape)}, name='${name}')`);
    }

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
          forwardLines.push(
            `${out} = Conv2D(filters=${Number(outChannels)}, kernel_size=${JSON.stringify(kernelSize)}, strides=${JSON.stringify(strides)}, padding='${padding}', use_bias=${node.inputs.length > 2 ? 'True' : 'False'}, name='${this.sanitize(node.name || out)}')(${inp})`,
          );
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
          forwardLines.push(
            `${out} = ${poolType}(pool_size=${JSON.stringify(kernelSize)}, strides=${JSON.stringify(strides)}, padding='${padding}', name='${this.sanitize(node.name || out)}')(${inp})`,
          );
          break;
        }
        case 'GlobalAveragePool': {
          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(
            `${out} = GlobalAveragePooling2D(name='${this.sanitize(node.name || out)}')(${inp})`,
          );
          break;
        }
        case 'Relu': {
          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(
            `${out} = Activation('relu', name='${this.sanitize(node.name || out)}')(${inp})`,
          );
          break;
        }
        case 'Flatten': {
          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(`${out} = Flatten(name='${this.sanitize(node.name || out)}')(${inp})`);
          break;
        }
        case 'Gemm':
        case 'MatMul': {
          const wShape = this.getShape(node.inputs[1]!);
          const outFeatures = wShape
            ? node.attributes['transB']?.value
              ? wShape[0]
              : wShape[1]
            : 10;
          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(
            `${out} = Dense(units=${Number(outFeatures)}, name='${this.sanitize(node.name || out)}')(${inp})`,
          );
          break;
        }
        case 'Add': {
          const inps = node.inputs.map((i) => this.sanitize(i)).join(', ');
          forwardLines.push(`${out} = Add(name='${this.sanitize(node.name || out)}')([${inps}])`);
          break;
        }
        case 'Softmax': {
          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(
            `${out} = Activation('softmax', name='${this.sanitize(node.name || out)}')(${inp})`,
          );
          break;
        }
        default: {
          const inps = node.inputs.map((i) => this.sanitize(i)).join(', ');
          forwardLines.push(
            `${out} = ${inps.split(', ')[0]}  # Fallback for ${String(node.opType)}`,
          );
        }
      }
    }

    if (forwardLines.length === 0) {
      forwardLines.push('pass');
    }
    for (const l of forwardLines) {
      lines.push(`    ${l}`);
    }

    const outputs = this.graph.outputs.map((o) => this.sanitize(o.name)).join(', ');
    if (outputs && inputNames.length > 0) {
      lines.push(`    model = Model(inputs=[${inputNames.join(', ')}], outputs=[${outputs}])`);
      lines.push(`    return model`);
    } else {
      lines.push(`    return None`);
    }

    return lines.join('\n') + '\n';
  }
}
