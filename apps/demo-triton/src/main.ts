/**
 * @fileoverview Triton Compiler Demo UI.
 */
import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { generateTriton } from '@onnx9000/triton-compiler';

/**
 * Initializes the Triton compiler demo UI.
 */
export function initTritonDemo(): void {
  const generateBtn = document.getElementById('generate-btn') as HTMLButtonElement | null;
  const out = document.getElementById('output') as HTMLElement | null;

  if (!generateBtn || !out) return;

  generateBtn.addEventListener('click', () => {
    out.innerText = 'Generating...';

    const g = new Graph('custom_fused_kernel');
    g.inputs.push(new ValueInfo('A', [1024], 'float32'));
    g.inputs.push(new ValueInfo('B', [1024], 'float32'));
    g.outputs.push(new ValueInfo('C', [1024], 'float32'));

    const addNode = new Node('Add', ['A', 'B'], ['C']);
    g.nodes.push(addNode);

    const code = generateTriton(g, { blockM: 128 });
    out.innerText = code;
  });
}

initTritonDemo();
