import { GraphMutator } from '@onnx9000/modifier';

/**
 * Initializes the GraphSurgeon demo.
 */
export function initGraphSurgeonDemo(): void {
  const mutateBtn = document.getElementById('mutate-btn') as HTMLButtonElement;
  const out = document.getElementById('surgeon-output') as HTMLElement;
  if (!mutateBtn || !out) return;

  mutateBtn.addEventListener('click', async () => {
    out.innerText = 'Initializing GraphSurgeon Mutator...\n';
    mutateBtn.disabled = true;

    try {
      const mockModel: any = {
        graph: {
          node: [
            { opType: 'Identity', name: 'id1', input: ['X'], output: ['Y'] },
            { opType: 'Relu', name: 'relu1', input: ['Y'], output: ['Z'] },
          ],
        },
      };

      out.innerText += '\nOriginal Graph:';
      out.innerText += `\n${JSON.stringify(mockModel.graph.node.map((n: any) => n.opType))}`;

      const mutator = new GraphMutator(mockModel);

      out.innerText += '\n\nApplying Mutations...';
      out.innerText += '\n- Removing Identity node';
      mutator.deleteNode('id1');

      out.innerText += '\n- Rewiring inputs';
      const reluNode = mockModel.graph.node.find((n: any) => n.name === 'relu1');
      if (reluNode) {
        reluNode.input[0] = 'X';
      }

      out.innerText += '\n\nMutated Graph:';
      out.innerText += `\n${JSON.stringify(mockModel.graph.node.filter((n: any) => n.name !== 'id1').map((n: any) => n.opType))}`;
      out.innerText += '\n\nSuccess! Graph structure modified.';
    } catch (e: any) {
      out.innerText += `\nError: ${e.message}`;
    } finally {
      mutateBtn.disabled = false;
    }
  });
}
initGraphSurgeonDemo();
