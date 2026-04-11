import { load, Graph, Node, Tensor } from '@onnx9000/core';

const dropZone = document.getElementById('drop-zone') as HTMLElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const codeArea = document.getElementById('code') as HTMLTextAreaElement;

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e: Event) => {
  const target = e.target as HTMLInputElement;
  if (target.files && target.files.length > 0) {
    processFile(target.files[0]);
  }
});

dropZone.addEventListener('dragover', (e: DragEvent) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e: DragEvent) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
    processFile(e.dataTransfer.files[0]);
  }
});

async function processFile(file: File) {
  if (!file.name.endsWith('.onnx')) {
    codeArea.value = '# Error: Please provide a valid .onnx file.';
    return;
  }
  codeArea.value = '# Loading and parsing ONNX AST...';
  try {
    const arrayBuffer = await file.arrayBuffer();
    const graph = await load(arrayBuffer);
    const code = generatePyTorchCode(graph);
    codeArea.value = code;
  } catch (_err) {
    const err = _err instanceof Error ? _err : new Error(String(_err));
    codeArea.value = `# Error during processing:\n${err.message || err.toString()}`;
  }
}

function generatePyTorchCode(graph: Graph): string {
  let code = `import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n`;
  code += `class GeneratedModel(nn.Module):\n`;
  code += `    def __init__(self):\n`;
  code += `        super(GeneratedModel, self).__init__()\n`;

  // Register parameters/buffers (mock logic)
  let initParams = '';
  for (const name of Object.keys(graph.tensors)) {
    const tensor = graph.tensors[name];
    if (tensor.isInitializer) {
      const shapeStr = tensor.shape.join(', ');
      initParams += `        self.register_buffer('${cleanName(name)}', torch.zeros(${shapeStr}))\n`;
    }
  }
  code += initParams || `        pass\n`;

  code += `\n    def forward(self, `;
  code += graph.inputs.map((i) => cleanName(i.name)).join(', ');
  code += `):\n`;

  const opEmitters: Record<string, (inputs: string[], outputs: string[], node: Node) => string> = {
    Relu: (inputs, outputs) => `        ${outputs[0]} = F.relu(${inputs[0]})\n`,
    Conv: (inputs, outputs, node) =>
      `        # Conv mapped to functional conv2d\n        ${outputs[0]} = F.conv2d(${inputs[0]}, self.${cleanName(node.inputs[1])})\n`,
    MatMul: (inputs, outputs) =>
      `        ${outputs[0]} = torch.matmul(${inputs[0]}, ${inputs[1]})\n`,
    Add: (inputs, outputs) => `        ${outputs[0]} = ${inputs[0]} + ${inputs[1]}\n`,
    Reshape: (inputs, outputs, node) =>
      `        ${outputs[0]} = torch.reshape(${inputs[0]}, self.${cleanName(node.inputs[1])}.tolist())\n`,
  };

  for (const node of graph.nodes) {
    code += `        # Node: ${node.name || node.opType}\n`;
    const inputs = node.inputs.map(cleanName);
    const outputs = node.outputs.map(cleanName);

    if (opEmitters[node.opType]) {
      code += opEmitters[node.opType](inputs, outputs, node);
    } else {
      const opName = node.opType.toLowerCase();
      code += `        ${outputs.join(', ')} = getattr(torch.ops.onnx, "${opName}")(${inputs.join(', ')})\n`;
    }
  }
  code += `\n        return `;
  code += graph.outputs.map((o) => cleanName(o.name)).join(', ');
  code += `\n`;
  return code;
}

function cleanName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_]/g, '_') || 'var_empty';
}
