import { Graph, ValueInfo, Node } from '@onnx9000/core';
import { GraphMutator } from '@onnx9000/modifier';

let currentGraph: Graph | null = null;
let mutator: GraphMutator | null = null;

const outputDiv = document.getElementById('output') as HTMLDivElement;

function logGraphState() {
  if (!currentGraph) {
    outputDiv.textContent = 'Graph not initialized.';
    return;
  }

  const inputs = currentGraph.inputs.map((i) => `${i.name} [${i.shape.join(',')}]`);
  const outputs = currentGraph.outputs.map((o) => `${o.name} [${o.shape.join(',')}]`);

  outputDiv.textContent = `Current Graph State:
Inputs: ${inputs.join(', ')}
Outputs: ${outputs.join(', ')}
Nodes: ${currentGraph.nodes.length}
`;
}

document.getElementById('btnInit')!.addEventListener('click', () => {
  currentGraph = new Graph('MockModel');

  // Add input
  const inp = new ValueInfo('input_0', [1, 3, 224, 224], 'float32');
  currentGraph.inputs.push(inp);

  // Add node
  const node = new Node('Relu', ['input_0'], ['output_0'], {}, 'relu_node');
  currentGraph.nodes.push(node);

  // Add output
  const out = new ValueInfo('output_0', [1, 3, 224, 224], 'float32');
  currentGraph.outputs.push(out);

  mutator = new GraphMutator(currentGraph);
  logGraphState();
});

document.getElementById('btnRename')!.addEventListener('click', () => {
  if (!mutator) return alert('Initialize graph first!');

  const oldName = (document.getElementById('oldInput') as HTMLInputElement).value;
  const newName = (document.getElementById('newInput') as HTMLInputElement).value;

  mutator.renameInput(oldName, newName);
  logGraphState();
});

document.getElementById('btnBatch')!.addEventListener('click', () => {
  if (!mutator || !currentGraph) return alert('Initialize graph first!');

  const batchStr = (document.getElementById('batchSize') as HTMLInputElement).value;
  const batchSize = isNaN(Number(batchStr)) ? batchStr : Number(batchStr);

  // Since GraphMutator doesn't have an explicit 'changeBatch' we manually update using overrideShape for inputs
  // or apply the standard headless JS equivalent

  for (const inp of currentGraph.inputs) {
    if (inp.shape.length > 0) {
      const newShape = [...inp.shape];
      newShape[0] = batchSize;
      mutator.overrideShape(inp.name, newShape, inp.dtype);
    }
  }
  for (const out of currentGraph.outputs) {
    if (out.shape.length > 0) {
      const newShape = [...out.shape];
      newShape[0] = batchSize;
      mutator.overrideShape(out.name, newShape, out.dtype);
    }
  }

  logGraphState();
});
