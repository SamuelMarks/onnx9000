/**
 * @fileoverview GraphSurgeon Demo UI.
 */
import { Graph, Node } from '@onnx9000/core';
import { GraphMutator } from '@onnx9000/modifier';

/**
 * Initializes the GraphSurgeon demo.
 */
export function initGraphSurgeonDemo(): void {
  const mutateBtn = document.getElementById('mutate-btn') as HTMLButtonElement | null;
  const out = document.getElementById('surgeon-output') as HTMLElement | null;
  if (!mutateBtn || !out) return;

  mutateBtn.addEventListener('click', () => {
    out.innerText = 'Initializing GraphSurgeon Mutator...';
    mutateBtn.disabled = true;

    const mockGraph = new Graph('surgeon_demo');
    mockGraph.nodes.push(
      new Node('Identity', ['X'], ['Y'], {}, 'id1'),
      new Node('Relu', ['Y'], ['Z'], {}, 'relu1'),
    );

    out.innerText += ' Original Graph: ';
    out.innerText += JSON.stringify(mockGraph.nodes.map((n: Node) => n.opType));

    const mutator = new GraphMutator(mockGraph);

    out.innerText += ' Applying Mutations... - Removing Identity node';
    mutator.removeNode('id1');

    out.innerText += ' - Rewiring inputs';
    for (const n of mockGraph.nodes) {
      if (n.name === 'relu1') {
        n.inputs[0] = 'X';
      }
    }

    out.innerText += ' Mutated Graph: ';
    out.innerText += JSON.stringify(mockGraph.nodes.map((n: Node) => n.opType));
    out.innerText += ' Success! Graph structure modified.';
    mutateBtn.disabled = false;
  });
}

initGraphSurgeonDemo();
