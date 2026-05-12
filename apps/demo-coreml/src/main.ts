import { Graph, Node } from '@onnx9000/core';
import { convertToCoreML } from '@onnx9000/coreml';

const convertBtn = document.getElementById('convert-btn') as HTMLButtonElement;
const out = document.getElementById('output') as HTMLElement;

convertBtn.addEventListener('click', () => {
  out.innerText = 'Converting...';

  // Build a mock ONNX graph
  const g = new Graph('mock_model');
  g.inputs.push({ name: 'input1', shape: [1, 3, 224, 224], type: null as any });
  g.outputs.push({ name: 'output1', shape: [1, 1000], type: null as any });

  const reluNode = new Node('Relu');
  reluNode.inputs = ['input1'];
  reluNode.outputs = ['output1'];
  g.nodes.push(reluNode);

  try {
    const milAst = convertToCoreML(g);

    // Custom replacer to handle bigint
    const jsonString = JSON.stringify(
      milAst,
      (key, value) => {
        if (typeof value === 'bigint') {
          return value.toString() + 'n';
        }
        return value;
      },
      2,
    );

    out.innerText = jsonString;
  } catch (e: any) {
    out.innerText = `Error: ${e.message}`;
  }
});
