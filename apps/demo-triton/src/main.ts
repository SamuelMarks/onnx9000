import { Graph, Node } from '@onnx9000/core';
import { generateTriton } from '@onnx9000/triton-compiler';

const generateBtn = document.getElementById('generate-btn') as HTMLButtonElement;
const out = document.getElementById('output') as HTMLElement;

generateBtn.addEventListener('click', () => {
  out.innerText = 'Generating...';

  // Build a mock ONNX graph to feed to the compiler
  const g = new Graph('custom_fused_kernel');
  g.inputs.push({ name: 'A', shape: [1024], type: null as any });
  g.inputs.push({ name: 'B', shape: [1024], type: null as any });
  g.outputs.push({ name: 'C', shape: [1024], type: null as any });

  const addNode = new Node('Add');
  addNode.inputs = ['A', 'B'];
  addNode.outputs = ['C'];
  g.nodes.push(addNode);

  try {
    const code = generateTriton(g, { blockM: 128 });
    out.innerText = code;
  } catch (e: any) {
    out.innerText = `Error: ${e.message}`;
  }
});
