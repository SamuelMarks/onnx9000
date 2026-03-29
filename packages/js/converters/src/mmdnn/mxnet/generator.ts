/* eslint-disable */
// @ts-nocheck
/* eslint-disable */
import { Graph } from '@onnx9000/core';

export class MXNetGenerator {
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
      'import mxnet as mx',
      'from mxnet import gluon',
      '',
      `class ${this.sanitize(this.graph.name || 'Model')}(gluon.HybridBlock):`,
      '    def __init__(self, **kwargs):',
      '        super().__init__(**kwargs)',
      '        with self.name_scope():',
    ];

    const initLines: string[] = [];
    const forwardLines: string[] = [];

    // Filter out initializers from inputs
    const trueInputs = this.graph.inputs.filter((inp) => !this.isInitializer(inp.name));
    const inputArgs = trueInputs.map((i) => this.sanitize(i.name)).join(', ');

    for (const node of this.graph.nodes) {
      const out = this.sanitize(node.outputs[0] || `out_${node.name || 'node'}`);

      switch (node.opType) {
        case 'Conv': {
          const wShape = this.getShape(node.inputs[1]!);
          const outChannels = wShape ? wShape[0] : 32;
          const kernelSize = wShape ? wShape.slice(2) : [3, 3];
          const strides = (node.attributes['strides']?.value as number[]) || [1, 1];
          const pads = (node.attributes['pads']?.value as number[]) || [0, 0, 0, 0];

          const layerName = this.sanitize(`conv_${node.name || out}`);
          initLines.push(
            `self.${layerName} = gluon.nn.Conv2D(channels=${Number(outChannels)}, kernel_size=${JSON.stringify(kernelSize)}, strides=${JSON.stringify(strides)}, padding=${JSON.stringify((pads as number[]).slice(0, 2))}, use_bias=${node.inputs.length > 2 ? 'True' : 'False'})`,
          );

          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(`${out} = self.${layerName}(${inp})`);
          break;
        }
        case 'MaxPool':
        case 'AveragePool': {
          const poolType = node.opType === 'MaxPool' ? 'MaxPool2D' : 'AvgPool2D';
          const kernelSize = (node.attributes['kernel_shape']?.value as number[]) || [2, 2];
          const strides = (node.attributes['strides']?.value as number[]) || [2, 2];
          const pads = (node.attributes['pads']?.value as number[]) || [0, 0, 0, 0];

          const layerName = this.sanitize(`pool_${node.name || out}`);
          initLines.push(
            `self.${layerName} = gluon.nn.${poolType}(pool_size=${JSON.stringify(kernelSize)}, strides=${JSON.stringify(strides)}, padding=${JSON.stringify((pads as number[]).slice(0, 2))})`,
          );

          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(`${out} = self.${layerName}(${inp})`);
          break;
        }
        case 'GlobalAveragePool': {
          const layerName = this.sanitize(`gap_${node.name || out}`);
          initLines.push(`self.${layerName} = gluon.nn.GlobalAvgPool2D()`);
          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(`${out} = self.${layerName}(${inp})`);
          break;
        }
        case 'Relu': {
          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(`${out} = mx.nd.relu(${inp})`);
          break;
        }
        case 'Flatten': {
          const layerName = this.sanitize(`flatten_${node.name || out}`);
          initLines.push(`self.${layerName} = gluon.nn.Flatten()`);
          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(`${out} = self.${layerName}(${inp})`);
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
          const layerName = this.sanitize(`dense_${node.name || out}`);

          initLines.push(`self.${layerName} = gluon.nn.Dense(units=${Number(outFeatures)})`);
          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(`${out} = self.${layerName}(${inp})`);
          break;
        }
        case 'Add': {
          const inp0 = this.sanitize(node.inputs[0]!);
          const inp1 = this.sanitize(node.inputs[1]!);
          forwardLines.push(`${out} = mx.nd.add(${inp0}, ${inp1})`);
          break;
        }
        case 'Softmax': {
          const inp = this.sanitize(node.inputs[0]!);
          const axis = node.attributes['axis'] != null ? Number(node.attributes['axis']) : -1;
          forwardLines.push(`${out} = mx.nd.softmax(${inp}, axis=${axis})`);
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

    if (initLines.length === 0) {
      initLines.push('pass');
    }
    for (const l of initLines) {
      lines.push(`            ${l}`);
    }

    lines.push('');
    lines.push(`    def hybrid_forward(self, F, inputs):`);
    if (inputArgs) {
      lines.push(`        ${inputArgs} = inputs`);
    }

    if (forwardLines.length === 0) {
      forwardLines.push('pass');
    }
    for (const l of forwardLines) {
      lines.push(`        ${l}`);
    }

    const outputs = this.graph.outputs.map((o) => this.sanitize(o.name)).join(', ');
    if (outputs) {
      lines.push(`        return ${outputs}`);
    } else {
      lines.push(`        return None`);
    }

    return lines.join('\n') + '\n';
  }
}
