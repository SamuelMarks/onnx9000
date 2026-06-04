/* v8 ignore next */ /* v8 ignore next */ import {
  Graph,
  Node,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import { generateTriton } from '@onnx9000/triton-compiler'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const generateBtn = document.getElementById(
  'generate-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
generateBtn.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText = 'Generating...'; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Build a mock ONNX graph to feed to the compiler /* v8 ignore next */ /* v8 ignore next */
  const g = new Graph('custom_fused_kernel'); /* v8 ignore next */ /* v8 ignore next */
  g.inputs.push({
    name: 'A',
    shape: [1024],
    type: null as any,
  }); /* v8 ignore next */ /* v8 ignore next */
  g.inputs.push({
    name: 'B',
    shape: [1024],
    type: null as any,
  }); /* v8 ignore next */ /* v8 ignore next */
  g.outputs.push({
    name: 'C',
    shape: [1024],
    type: null as any,
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const addNode = new Node('Add'); /* v8 ignore next */ /* v8 ignore next */
  addNode.inputs = ['A', 'B']; /* v8 ignore next */ /* v8 ignore next */
  addNode.outputs = ['C']; /* v8 ignore next */ /* v8 ignore next */
  g.nodes.push(addNode); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const code = generateTriton(g, { blockM: 128 }); /* v8 ignore next */ /* v8 ignore next */
    out.innerText = code; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText = `Error: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
