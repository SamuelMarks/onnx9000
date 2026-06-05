import { Graph } from '../ir/graph.js';
import { Node } from '../ir/node.js';

export function cleanName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_]/g, '_') || 'var_empty';
}

export class ONNXToPyTorchVisitor {
  private graph: Graph;

  constructor(graph: Graph) {
    this.graph = graph;
  }

  generate(): string {
    let code = `import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n`;
    code += `class GeneratedModel(nn.Module):\n`;
    code += `    def __init__(self):\n`;
    code += `        super(GeneratedModel, self).__init__()\n`;

    // Register parameters/buffers
    let initParams = '';
    for (const name of Object.keys(this.graph.tensors)) {
      const tensor = this.graph.tensors[name];
      if (!tensor) continue;
      if (tensor.isInitializer) {
        const shapeStr = tensor.shape.join(', ');
        initParams += `        self.register_buffer('${cleanName(name)}', torch.zeros(${shapeStr}))\n`;
      }
    }
    code += initParams || `        pass\n`;

    code += `\n    def forward(self`;
    if (this.graph.inputs.length > 0) {
      code += `, ` + this.graph.inputs.map((i) => cleanName(i.name)).join(', ');
    }
    code += `):\n`;

    const opEmitters: Record<string, (inputs: string[], outputs: string[], node: Node) => string> =
      {
        Relu: (inputs, outputs) => `        ${outputs[0] || ''} = F.relu(${inputs[0] || ''})\n`,
        Conv: (inputs, outputs, node) =>
          `        # Conv mapped to functional conv2d\n        ${outputs[0] || ''} = F.conv2d(${inputs[0] || ''}, self.${cleanName(node.inputs[1] || '')})\n`,
        MatMul: (inputs, outputs) =>
          `        ${outputs[0] || ''} = torch.matmul(${inputs[0] || ''}, ${inputs[1] || ''})\n`,
        Add: (inputs, outputs) =>
          `        ${outputs[0] || ''} = ${inputs[0] || ''} + ${inputs[1] || ''}\n`,
        Reshape: (inputs, outputs, node) =>
          `        ${outputs[0] || ''} = torch.reshape(${inputs[0] || ''}, self.${cleanName(node.inputs[1] || '')}.tolist())\n`,
      };

    if (this.graph.nodes.length === 0) {
      code += `        pass\n`;
    }

    for (const node of this.graph.nodes) {
      code += `        # Node: ${node.name || node.opType}\n`;
      const inputs = node.inputs.map(cleanName);
      const outputs = node.outputs.map(cleanName);

      const emitter = opEmitters[node.opType];
      if (emitter) {
        code += emitter(inputs, outputs, node);
      } else {
        const opName = node.opType.toLowerCase();
        code += `        ${outputs.join(', ')} = getattr(torch.ops.onnx, "${opName}")(${inputs.join(', ')})\n`;
      }
    }
    code += `\n        return `;
    if (this.graph.outputs.length > 0) {
      code += this.graph.outputs.map((o) => cleanName(o.name)).join(', ');
    } else {
      code += `None`;
    }
    code += `\n`;
    return code;
  }
}
