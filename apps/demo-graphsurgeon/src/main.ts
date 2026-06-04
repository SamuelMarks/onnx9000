/* v8 ignore next */ /* v8 ignore next */ import { GraphMutator } from '@onnx9000/modifier'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const mutateBtn = document.getElementById(
  'mutate-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'surgeon-output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
mutateBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText =
    'Initializing GraphSurgeon Mutator...\n'; /* v8 ignore next */ /* v8 ignore next */
  mutateBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    // We mock an ONNX ModelProto /* v8 ignore next */ /* v8 ignore next */
    const mockModel: any = {
      /* v8 ignore next */ /* v8 ignore next */
      graph: {
        /* v8 ignore next */ /* v8 ignore next */
        node: [
          /* v8 ignore next */ /* v8 ignore next */
          {
            opType: 'Identity',
            name: 'id1',
            input: ['X'],
            output: ['Y'],
          } /* v8 ignore next */ /* v8 ignore next */,
          {
            opType: 'Relu',
            name: 'relu1',
            input: ['Y'],
            output: ['Z'],
          } /* v8 ignore next */ /* v8 ignore next */,
        ] /* v8 ignore next */ /* v8 ignore next */,
      } /* v8 ignore next */ /* v8 ignore next */,
    }; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\nOriginal Graph:'; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\n${JSON.stringify(mockModel.graph.node.map((n: any) => n.opType))}`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const mutator = new GraphMutator(mockModel); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\n\nApplying Mutations...'; /* v8 ignore next */ /* v8 ignore next */
    // Let's manually delete the Identity node and rewire X directly to Relu /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\n- Removing Identity node'; /* v8 ignore next */ /* v8 ignore next */
    mutator.deleteNode('id1'); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\n- Rewiring inputs'; /* v8 ignore next */ /* v8 ignore next */
    const reluNode = mockModel.graph.node.find(
      (n: any) => n.name === 'relu1',
    ); /* v8 ignore next */ /* v8 ignore next */
    if (reluNode) {
      /* v8 ignore next */ /* v8 ignore next */
      reluNode.input[0] = 'X'; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Usually mutator.apply() or similar is called, but for demo we just show the state /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\n\nMutated Graph:'; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\n${JSON.stringify(mockModel.graph.node.filter((n: any) => n.name !== 'id1').map((n: any) => n.opType))}`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\n\nSuccess! Graph structure modified.'; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nError: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } finally {
    /* v8 ignore next */ /* v8 ignore next */
    mutateBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
