/**
 * @fileoverview TVM Relay Demo UI.
 */
import { Graph, Node, ValueInfo } from '@onnx9000/core';

/**
 * Initializes the TVM demo UI.
 */
export function initTvmDemo(): void {
  const convertBtn = document.getElementById('convert-btn') as HTMLButtonElement | null;
  const out = document.getElementById('output') as HTMLElement | null;

  if (!convertBtn || !out) return;

  convertBtn.addEventListener('click', () => {
    out.innerText = 'Converting...';

    const g = new Graph('mock_model');
    g.inputs.push(new ValueInfo('input1', [1, 3, 224, 224], 'float32'));
    g.outputs.push(new ValueInfo('output1', [1, 1000], 'float32'));

    const reluNode = new Node('Relu', ['input1'], ['output1']);
    g.nodes.push(reluNode);

    out.innerText = `#[version = "0.0.5"]
def @main(%input1: Tensor[(1, 3, 224, 224), float32]) -> Tensor[(1, 1000), float32] {
  nn.relu(%input1)
}`;
  });
}

initTvmDemo();
