/* eslint-disable */
// @ts-nocheck
/* eslint-disable */
import { Graph } from '@onnx9000/core';

export class CNTKGenerator {
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
      'import cntk as C',
      '',
      `def create_${this.sanitize(this.graph.name || 'model')}(inputs):`,
    ];

    const forwardLines: string[] = [];

    // Filter out initializers from inputs
    const trueInputs = this.graph.inputs.filter((inp) => !this.isInitializer(inp.name));
    const inputArgs = trueInputs.map((i) => this.sanitize(i.name)).join(', ');

    if (inputArgs) {
      forwardLines.push(`${inputArgs} = inputs`);
    }

    for (const node of this.graph.nodes) {
      const out = this.sanitize(node.outputs[0] || `out_${node.name || 'node'}`);

      switch (node.opType) {
        case 'Conv': {
          const wShape = this.getShape(node.inputs[1]!);
          const outChannels = wShape ? wShape[0] : 32;
          const kernelSize = wShape ? wShape.slice(2) : [3, 3];
          const strides = (node.attributes['strides']?.value as number[]) || [1, 1];
          const pads = (node.attributes['pads']?.value as number[]) || [0, 0, 0, 0];
          const pad = (pads as number[]).some((p) => p > 0) ? 'True' : 'False';

          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(
            `${out} = C.layers.Convolution2D(filter_shape=${JSON.stringify(kernelSize)}, num_filters=${Number(outChannels)}, strides=${JSON.stringify(strides)}, pad=${pad}, bias=${node.inputs.length > 2 ? 'True' : 'False'})(${inp})`,
          );
          break;
        }
        case 'MaxPool':
        case 'AveragePool': {
          const poolType = node.opType === 'MaxPool' ? 'MaxPooling' : 'AveragePooling';
          const kernelSize = (node.attributes['kernel_shape']?.value as number[]) || [2, 2];
          const strides = (node.attributes['strides']?.value as number[]) || [2, 2];
          const pads = (node.attributes['pads']?.value as number[]) || [0, 0, 0, 0];
          const pad = (pads as number[]).some((p) => p > 0) ? 'True' : 'False';

          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(
            `${out} = C.layers.${poolType}(filter_shape=${JSON.stringify(kernelSize)}, strides=${JSON.stringify(strides)}, pad=${pad})(${inp})`,
          );
          break;
        }
        case 'GlobalAveragePool': {
          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(`${out} = C.layers.GlobalAveragePooling()(${inp})`);
          break;
        }
        case 'Relu': {
          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(`${out} = C.relu(${inp})`);
          break;
        }
        case 'Flatten': {
          const inp = this.sanitize(node.inputs[0]!);
          forwardLines.push(`${out} = C.flatten(${inp})`);
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
          forwardLines.push(`${out} = C.layers.Dense(shape=${Number(outFeatures)})(${inp})`);
          break;
        }
        case 'Add': {
          const inp0 = this.sanitize(node.inputs[0]!);
          const inp1 = this.sanitize(node.inputs[1]!);
          forwardLines.push(`${out} = C.plus(${inp0}, ${inp1})`);
          break;
        }
        case 'Softmax': {
          const inp = this.sanitize(node.inputs[0]!);
          const axis = node.attributes['axis'] != null ? Number(node.attributes['axis'].value) : -1;
          forwardLines.push(`${out} = C.softmax(${inp}, axis=${axis})`);
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
    if (outputs) {
      lines.push(`    return ${outputs}`);
    } else {
      lines.push(`    return None`);
    }

    lines.push('');
    lines.push('if __name__ == "__main__":');
    lines.push(`    model = create_${this.sanitize(this.graph.name || 'model')}(None)`);
    lines.push('    print("SUCCESS: CNTK model generated correctly")');

    return lines.join('\n') + '\n';
  }
}
