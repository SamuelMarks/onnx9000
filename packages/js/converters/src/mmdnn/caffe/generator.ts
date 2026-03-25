/* eslint-disable */
import { Graph } from '@onnx9000/core';

export class CaffeGenerator {
  graph: Graph;

  constructor(graph: Graph) {
    this.graph = graph;
  }

  private sanitize(name: string): string {
    if (!name) return 'unnamed';
    return name.replace(/[^a-zA-Z0-9_]/g, '_');
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
    const lines = [`name: "${this.graph.name || 'Model'}"`];

    // Filter out initializers from inputs
    const trueInputs = this.graph.inputs.filter((inp) => !this.isInitializer(inp.name));

    for (const inp of trueInputs) {
      const shape = this.getShape(inp.name) || [1, 3, 224, 224];
      lines.push(`layer {`);
      lines.push(`  name: "${this.sanitize(inp.name)}"`);
      lines.push(`  type: "Input"`);
      lines.push(`  top: "${this.sanitize(inp.name)}"`);
      lines.push(`  input_param { shape: { ${shape.map((d) => `dim: ${d}`).join(' ')} } }`);
      lines.push(`}`);
    }

    for (const node of this.graph.nodes) {
      const out = this.sanitize(node.outputs[0] || `out_${node.name || 'node'}`);
      const inps = node.inputs.map((i) => this.sanitize(i));

      lines.push(`layer {`);
      lines.push(`  name: "${this.sanitize(node.name || out)}"`);

      switch (node.opType) {
        case 'Conv': {
          lines.push(`  type: "Convolution"`);
          for (const inp of inps) {
            if (!this.isInitializer(inp)) lines.push(`  bottom: "${inp}"`);
          }
          lines.push(`  top: "${out}"`);
          const wShape = this.getShape(node.inputs[1]!);
          const outChannels = wShape ? wShape[0] : 32;
          const kernelSize = wShape ? wShape.slice(2) : [3, 3];
          const strides = (node.attributes['strides']?.value as number[]) || [1, 1];

          lines.push(`  convolution_param {`);
          lines.push(`    num_output: ${Number(outChannels)}`);
          if (kernelSize.length === 2) {
            lines.push(`    kernel_h: ${kernelSize[0]}`);
            lines.push(`    kernel_w: ${kernelSize[1]}`);
          }
          if (strides.length === 2) {
            lines.push(`    stride_h: ${strides[0]}`);
            lines.push(`    stride_w: ${strides[1]}`);
          }
          lines.push(`  }`);
          break;
        }
        case 'MaxPool':
        case 'AveragePool': {
          lines.push(`  type: "Pooling"`);
          for (const inp of inps) {
            if (!this.isInitializer(inp)) lines.push(`  bottom: "${inp}"`);
          }
          lines.push(`  top: "${out}"`);

          const poolMethod = node.opType === 'MaxPool' ? 'MAX' : 'AVE';
          const kernelSize = (node.attributes['kernel_shape']?.value as number[]) || [2, 2];
          const strides = (node.attributes['strides']?.value as number[]) || [2, 2];

          lines.push(`  pooling_param {`);
          lines.push(`    pool: ${poolMethod}`);
          if (kernelSize.length === 2) {
            lines.push(`    kernel_h: ${kernelSize[0]}`);
            lines.push(`    kernel_w: ${kernelSize[1]}`);
          }
          if (strides.length === 2) {
            lines.push(`    stride_h: ${strides[0]}`);
            lines.push(`    stride_w: ${strides[1]}`);
          }
          lines.push(`  }`);
          break;
        }
        case 'Relu': {
          lines.push(`  type: "ReLU"`);
          lines.push(`  bottom: "${inps[0]}"`);
          lines.push(`  top: "${out}"`);
          break;
        }
        case 'Flatten': {
          lines.push(`  type: "Flatten"`);
          lines.push(`  bottom: "${inps[0]}"`);
          lines.push(`  top: "${out}"`);
          break;
        }
        case 'Gemm':
        case 'MatMul': {
          lines.push(`  type: "InnerProduct"`);
          lines.push(`  bottom: "${inps[0]}"`);
          lines.push(`  top: "${out}"`);
          const wShape = this.getShape(node.inputs[1]!);
          const outFeatures = wShape
            ? node.attributes['transB']?.value
              ? wShape[0]
              : wShape[1]
            : 10;
          lines.push(`  inner_product_param {`);
          lines.push(`    num_output: ${Number(outFeatures)}`);
          lines.push(`  }`);
          break;
        }
        case 'Softmax': {
          lines.push(`  type: "Softmax"`);
          lines.push(`  bottom: "${inps[0]}"`);
          lines.push(`  top: "${out}"`);
          break;
        }
        case 'GlobalAveragePool': {
          lines.push(`  type: "Pooling"`);
          lines.push(`  bottom: "${inps[0]}"`);
          lines.push(`  top: "${out}"`);
          lines.push(`  pooling_param {`);
          lines.push(`    pool: AVE`);
          lines.push(`    global_pooling: true`);
          lines.push(`  }`);
          break;
        }
        case 'Add': {
          lines.push(`  type: "Eltwise"`);
          for (const inp of inps) {
            if (!this.isInitializer(inp)) lines.push(`  bottom: "${inp}"`);
          }
          lines.push(`  top: "${out}"`);
          lines.push(`  eltwise_param { operation: SUM }`);
          break;
        }
        default: {
          lines.push(`  type: "Dummy${String(node.opType)}"`);
          for (const inp of inps) {
            if (!this.isInitializer(inp)) lines.push(`  bottom: "${inp}"`);
          }
          lines.push(`  top: "${out}"`);
        }
      }
      lines.push(`}`);
    }

    return lines.join('\n') + '\n';
  }
}
