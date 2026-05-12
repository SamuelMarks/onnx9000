import { Graph, Node } from '@onnx9000/core';

const convertBtn = document.getElementById('convert-btn') as HTMLButtonElement;
const out = document.getElementById('output') as HTMLElement;

convertBtn.addEventListener('click', () => {
  out.innerText = 'Converting...';

  const g = new Graph('mock_model');
  g.inputs.push({ name: 'input1', shape: [1, 3, 224, 224], type: null as any });
  g.outputs.push({ name: 'output1', shape: [1, 1000], type: null as any });

  const reluNode = new Node('Relu');
  reluNode.inputs = ['input1'];
  reluNode.outputs = ['output1'];
  g.nodes.push(reluNode);

  out.innerText = `#[version = "0.0.5"]\ndef @main(%input1: Tensor[(1, 3, 224, 224), float32]) -> Tensor[(1, 1000), float32] {\n  nn.relu(%input1)\n}`;
});
