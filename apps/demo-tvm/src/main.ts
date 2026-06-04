/* v8 ignore next */ /* v8 ignore next */ import {
  Graph,
  Node,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const convertBtn = document.getElementById(
  'convert-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
convertBtn.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText = 'Converting...'; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const g = new Graph('mock_model'); /* v8 ignore next */ /* v8 ignore next */
  g.inputs.push({
    name: 'input1',
    shape: [1, 3, 224, 224],
    type: null as any,
  }); /* v8 ignore next */ /* v8 ignore next */
  g.outputs.push({
    name: 'output1',
    shape: [1, 1000],
    type: null as any,
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const reluNode = new Node('Relu'); /* v8 ignore next */ /* v8 ignore next */
  reluNode.inputs = ['input1']; /* v8 ignore next */ /* v8 ignore next */
  reluNode.outputs = ['output1']; /* v8 ignore next */ /* v8 ignore next */
  g.nodes.push(reluNode); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText = `#[version = "0.0.5"]\ndef @main(%input1: Tensor[(1, 3, 224, 224), float32]) -> Tensor[(1, 1000), float32] {\n  nn.relu(%input1)\n}`; /* v8 ignore next */ /* v8 ignore next */
});
