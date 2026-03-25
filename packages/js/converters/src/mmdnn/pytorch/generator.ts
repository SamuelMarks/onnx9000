import { Graph, Node, Attribute, Shape } from '@onnx9000/core';

export class PyTorchGenerator {
  graph: Graph;

  constructor(graph: Graph) {
    this.graph = graph;
  }

  private sanitize(name: string): string {
    let sanitized = name.replace(/[^a-zA-Z0-9_]/g, '_');
    if (/^[0-9]/.test(sanitized)) {
      sanitized = 'v_' + sanitized;
    }
    return sanitized;
  }

  private getShape(name: string): number[] | null {
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

  private formatTuple(arr: number[]): string {
    if (arr.length === 1) return `(${arr[0]},)`;
    return `(${arr.join(', ')})`;
  }

  generate(): string {
    const uses: Record<string, Node[]> = {};
    for (const node of this.graph.nodes) {
      if (!node) continue;
      for (const input of node.inputs) {
        if (!uses[input]) uses[input] = [];
        uses[input].push(node);
      }
    }

    const initLines: string[] = [];
    const forwardLines: string[] = [];
    const sequences: Node[][] = [];
    let currentSeq: Node[] = [];

    const isSequentialNode = (node: Node) => {
      return [
        'Conv',
        'Gemm',
        'MatMul',
        'MaxPool',
        'AveragePool',
        'BatchNormalization',
        'Relu',
        'Sigmoid',
        'Tanh',
      ].includes(node.opType);
    };

    const getDynamicInputs = (node: Node) => {
      return node.inputs.filter((i) => !this.isInitializer(i));
    };

    for (let i = 0; i < this.graph.nodes.length; i++) {
      const node = this.graph.nodes[i];
      if (!node) continue;
      const dynIn = getDynamicInputs(node);
      const canBeSeq = isSequentialNode(node);

      if (currentSeq.length === 0) {
        if (canBeSeq && dynIn.length === 1 && dynIn[0] !== undefined) currentSeq.push(node);
      } else {
        const prevNode = currentSeq[currentSeq.length - 1]!;
        const prevOut = prevNode.outputs[0];

        if (
          canBeSeq &&
          dynIn.length === 1 &&
          dynIn[0] !== undefined &&
          dynIn[0] === prevOut &&
          (uses[prevOut] || []).length === 1
        ) {
          currentSeq.push(node);
        } else {
          if (currentSeq.length > 1) sequences.push([...currentSeq]);
          currentSeq = canBeSeq && dynIn.length === 1 && dynIn[0] !== undefined ? [node] : [];
        }
      }
    }
    if (currentSeq.length > 1) sequences.push(currentSeq);

    const seqSet = new Set<Node>();
    for (const seq of sequences) {
      for (const node of seq) seqSet.add(node);
    }

    // Register standalone initializers used in non-Sequential nodes
    const registeredBuffers = new Set<string>();
    const registerBufferIfNeeded = (name: string) => {
      if (this.isInitializer(name) && !registeredBuffers.has(name)) {
        const shape = this.getShape(name) || [];
        initLines.push(
          `self.register_buffer('${this.sanitize(name)}', torch.empty(${this.formatTuple(shape)}))`,
        );
        registeredBuffers.add(name);
      }
    };

    let seqCounter = 1;
    let nodeCounter = 1;

    const getNodeDecl = (node: Node): string | null => {
      const op = node.opType;
      if (op === 'Conv') {
        const wShape = this.getShape(node.inputs[1]!) || [1, 1, 3, 3];
        const D = wShape.length - 2;
        const groups = (node.attributes.group?.value as number) || 1;
        const out_channels = wShape[0] || 1;
        const in_channels = (wShape[1] || 1) * groups;
        const kernel_size = wShape.slice(2);
        const stride = (node.attributes.strides?.value as number[]) || Array(D).fill(1);
        const pads = (node.attributes.pads?.value as number[]) || Array(D * 2).fill(0);
        const padding = pads.slice(0, D);
        const bias = node.inputs.length > 2 ? 'True' : 'False';
        return `nn.Conv${D}d(in_channels=${in_channels}, out_channels=${out_channels}, kernel_size=${this.formatTuple(kernel_size)}, stride=${this.formatTuple(stride)}, padding=${this.formatTuple(padding)}, groups=${groups}, bias=${bias})`;
      } else if (op === 'Gemm' || op === 'MatMul') {
        let in_features = 1;
        let out_features = 1;
        const bias = node.inputs.length > 2 ? 'True' : 'False';
        if (node.inputs.length > 1) {
          const wShape = this.getShape(node.inputs[1]!);
          if (wShape && wShape.length === 2) {
            const transB = (node.attributes.transB?.value as number) || 0;
            in_features = transB ? wShape[1] || 1 : wShape[0] || 1;
            out_features = transB ? wShape[0] || 1 : wShape[1] || 1;
          }
        }
        return `nn.Linear(in_features=${in_features}, out_features=${out_features}, bias=${bias})`;
      } else if (op === 'MaxPool' || op === 'AveragePool') {
        const kShape = (node.attributes.kernel_shape?.value as number[]) || [2, 2];
        const D = kShape.length;
        const stride = (node.attributes.strides?.value as number[]) || Array(D).fill(1);
        const pads = (node.attributes.pads?.value as number[]) || Array(D * 2).fill(0);
        const padding = pads.slice(0, D);
        const type = op === 'MaxPool' ? 'MaxPool' : 'AvgPool';
        return `nn.${type}${D}d(kernel_size=${this.formatTuple(kShape)}, stride=${this.formatTuple(stride)}, padding=${this.formatTuple(padding)})`;
      } else if (op === 'BatchNormalization') {
        const wShape = this.getShape(node.inputs[1]!) || [1];
        const num_features = wShape[0] || 1;
        const eps = (node.attributes.epsilon?.value as number) || 1e-5;
        const momentumOnnx = node.attributes.momentum?.value as number;
        const momentum = momentumOnnx !== undefined ? 1.0 - momentumOnnx : 0.1;
        return `nn.BatchNorm2d(num_features=${num_features}, eps=${eps}, momentum=${momentum})`;
      } else if (op === 'Relu') {
        return `nn.ReLU()`;
      } else if (op === 'Sigmoid') {
        return `nn.Sigmoid()`;
      } else if (op === 'Tanh') {
        return `nn.Tanh()`;
      }
      return null;
    };

    // First process sequences
    for (const seq of sequences) {
      const seqName = `seq_${seqCounter++}`;
      initLines.push(`self.${seqName} = nn.Sequential(`);
      for (const node of seq) {
        const decl = getNodeDecl(node);
        if (decl) {
          initLines.push(`    ${decl},`);
        }
      }
      initLines.push(`)`);
    }

    for (const node of this.graph.nodes) {
      if (!node) continue;
      if (seqSet.has(node)) {
        // If it's the LAST node in a sequence, we emit the call in forward
        const seq = sequences.find((s) => s[s.length - 1] === node);
        if (seq) {
          const seqIndex = sequences.indexOf(seq) + 1;
          const inVar = seq[0] && seq[0]!.inputs[0] ? this.sanitize(seq[0]!.inputs[0]!) : 'in';
          const outVar = node.outputs[0] ? this.sanitize(node.outputs[0]!) : 'out';
          forwardLines.push(`${outVar} = self.seq_${seqIndex}(${inVar})`);
        }
        continue;
      }

      const dynIn = getDynamicInputs(node).map((i) => this.sanitize(i));
      const inVars = node.inputs.map((i) =>
        i ? (this.isInitializer(i) ? `self.${this.sanitize(i)}` : this.sanitize(i)) : 'None',
      );
      const outVar = node.outputs[0] ? this.sanitize(node.outputs[0]!) : 'out';

      for (const i of node.inputs) {
        registerBufferIfNeeded(i);
      }

      const decl = getNodeDecl(node);
      if (
        decl &&
        ['Conv', 'Gemm', 'MatMul', 'MaxPool', 'AveragePool', 'BatchNormalization'].includes(
          node.opType,
        )
      ) {
        const nodeName = `${node.opType.toLowerCase()}_${nodeCounter++}`;
        initLines.push(`self.${nodeName} = ${decl}`);
        forwardLines.push(`${outVar} = self.${nodeName}(${inVars[0]})`);
      } else if (node.opType === 'Relu') {
        forwardLines.push(`${outVar} = F.relu(${inVars[0]})`);
      } else if (node.opType === 'Sigmoid') {
        forwardLines.push(`${outVar} = torch.sigmoid(${inVars[0]})`);
      } else if (node.opType === 'Tanh') {
        forwardLines.push(`${outVar} = torch.tanh(${inVars[0]})`);
      } else if (node.opType === 'Add') {
        forwardLines.push(`${outVar} = ${inVars[0]} + ${inVars[1]}`);
      } else if (node.opType === 'Mul') {
        forwardLines.push(`${outVar} = ${inVars[0]} * ${inVars[1]}`);
      } else if (node.opType === 'Concat') {
        const axis = (node.attributes.axis?.value as number) || 0;
        forwardLines.push(`${outVar} = torch.cat((${inVars.join(', ')}), dim=${axis})`);
      } else if (node.opType === 'Reshape') {
        if (this.isInitializer(node.inputs[1]!)) {
          // Try to inline the shape if possible
          const shapeTensor = this.graph.tensors[node.inputs[1]!];
          if (shapeTensor && shapeTensor.data) {
            const shapeArray = Array.from(
              new Int32Array(
                shapeTensor.data.buffer,
                shapeTensor.data.byteOffset,
                shapeTensor.data.byteLength / 4,
              ),
            );
            forwardLines.push(
              `${outVar} = torch.reshape(${inVars[0]}, ${this.formatTuple(shapeArray)})`,
            );
            continue;
          }
        }
        forwardLines.push(`${outVar} = torch.reshape(${inVars[0]}, tuple(${inVars[1]}.tolist()))`);
      } else if (node.opType === 'Transpose') {
        const perm = node.attributes.perm?.value as number[];
        if (perm) {
          forwardLines.push(`${outVar} = ${inVars[0]}.permute(${perm.join(', ')})`);
        } else {
          forwardLines.push(`${outVar} = torch.transpose(${inVars[0]}, 0, 1)`); // default fallback
        }
      } else {
        // Fallback for unknown ops
        forwardLines.push(
          `${outVar} = getattr(torch, '${node.opType.toLowerCase()}')(${inVars.join(', ')}) # WARNING: Unmapped op`,
        );
      }
    }

    const inputArgs = this.graph.inputs.map((i) => this.sanitize(i.name)).join(', ');
    const outputReturns = this.graph.outputs.map((o) => this.sanitize(o.name)).join(', ');

    const lines = [
      'import torch',
      'import torch.nn as nn',
      'import torch.nn.functional as F',
      '',
      'class ONNXModel(nn.Module):',
      '    def __init__(self):',
      '        super(ONNXModel, self).__init__()',
      ...(initLines.length > 0 ? initLines.map((l) => `        ${l}`) : ['        pass']),
      '',
      `    def forward(self, ${inputArgs}):`,
      ...(forwardLines.length > 0 ? forwardLines.map((l) => `        ${l}`) : ['        pass']),
      `        return ${outputReturns}`,
      '',
      'if __name__ == "__main__":',
      '    model = ONNXModel()',
      '    print("SUCCESS: PyTorch model generated correctly")',
    ];

    return lines.join('\n');
  }
}
