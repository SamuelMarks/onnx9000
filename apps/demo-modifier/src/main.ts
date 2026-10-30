/**
 * @fileoverview Graph Modifier Demo UI.
 */
import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { GraphMutator } from '@onnx9000/modifier';

/**
 * Initializes the modifier demo UI.
 */
export function initModifierDemo(): void {
  let currentGraph: Graph | null = null;
  let mutator: GraphMutator | null = null;

  const outputDiv = document.getElementById('output') as HTMLDivElement | null;
  const btnInit = document.getElementById('btnInit');
  const btnRename = document.getElementById('btnRename');
  const btnBatch = document.getElementById('btnBatch');

  if (!outputDiv || !btnInit || !btnRename || !btnBatch) return;

  const validOutputDiv = outputDiv;

  function logGraphState(graph: Graph): void {
    const inputs = graph.inputs.map((i) => `${i.name} [${i.shape.join(',')}]`);
    const outputs = graph.outputs.map((o) => `${o.name} [${o.shape.join(',')}]`);

    validOutputDiv.textContent = `Current Graph State: 
Inputs: ${inputs.join(', ')} 
Outputs: ${outputs.join(', ')} 
Nodes: ${graph.nodes.length} 
`;
  }

  btnInit.addEventListener('click', () => {
    currentGraph = new Graph('MockModel');
    const inp = new ValueInfo('input_0', [1, 3, 224, 224], 'float32');
    currentGraph.inputs.push(inp);
    const node = new Node('Relu', ['input_0'], ['output_0'], {}, 'relu_node');
    currentGraph.nodes.push(node);
    const out = new ValueInfo('output_0', [1, 3, 224, 224], 'float32');
    currentGraph.outputs.push(out);
    mutator = new GraphMutator(currentGraph);
    logGraphState(currentGraph);
  });

  btnRename.addEventListener('click', () => {
    if (!mutator || !currentGraph) {
      alert('Initialize graph first!');
      return;
    }
    const oldName = (document.getElementById('oldInput') as HTMLInputElement).value;
    const newName = (document.getElementById('newInput') as HTMLInputElement).value;
    mutator.renameInput(oldName, newName);
    logGraphState(currentGraph);
  });

  btnBatch.addEventListener('click', () => {
    if (!mutator || !currentGraph) {
      alert('Initialize graph first!');
      return;
    }
    const batchStr = (document.getElementById('batchSize') as HTMLInputElement).value;
    const batchSize = Number.isNaN(Number(batchStr)) ? batchStr : Number(batchStr);

    for (const inp of currentGraph.inputs) {
      const newShape = [...inp.shape];
      newShape[0] = batchSize;
      mutator.overrideShape(inp.name, newShape, inp.dtype);
    }
    for (const out of currentGraph.outputs) {
      const newShape = [...out.shape];
      newShape[0] = batchSize;
      mutator.overrideShape(out.name, newShape, out.dtype);
    }
    logGraphState(currentGraph);
  });
}

initModifierDemo();
