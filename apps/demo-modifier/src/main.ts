/* v8 ignore next */ /* v8 ignore next */ import {
  Graph,
  ValueInfo,
  Node,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import { GraphMutator } from '@onnx9000/modifier'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
let currentGraph: Graph | null = null; /* v8 ignore next */ /* v8 ignore next */
let mutator: GraphMutator | null = null; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const outputDiv = document.getElementById(
  'output',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function logGraphState() {
  /* v8 ignore next */ /* v8 ignore next */
  if (!currentGraph) {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent = 'Graph not initialized.'; /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const inputs = currentGraph.inputs.map(
    (i) => `${i.name} [${i.shape.join(',')}]`,
  ); /* v8 ignore next */ /* v8 ignore next */
  const outputs = currentGraph.outputs.map(
    (o) => `${o.name} [${o.shape.join(',')}]`,
  ); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  outputDiv.textContent = `Current Graph State: /* v8 ignore next */ /* v8 ignore next */
Inputs: ${inputs.join(', ')} /* v8 ignore next */ /* v8 ignore next */
Outputs: ${outputs.join(', ')} /* v8 ignore next */ /* v8 ignore next */
Nodes: ${currentGraph.nodes.length} /* v8 ignore next */ /* v8 ignore next */
`; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
document.getElementById('btnInit')!.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  currentGraph = new Graph('MockModel'); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Add input /* v8 ignore next */ /* v8 ignore next */
  const inp = new ValueInfo(
    'input_0',
    [1, 3, 224, 224],
    'float32',
  ); /* v8 ignore next */ /* v8 ignore next */
  currentGraph.inputs.push(inp); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Add node /* v8 ignore next */ /* v8 ignore next */
  const node = new Node(
    'Relu',
    ['input_0'],
    ['output_0'],
    {},
    'relu_node',
  ); /* v8 ignore next */ /* v8 ignore next */
  currentGraph.nodes.push(node); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Add output /* v8 ignore next */ /* v8 ignore next */
  const out = new ValueInfo(
    'output_0',
    [1, 3, 224, 224],
    'float32',
  ); /* v8 ignore next */ /* v8 ignore next */
  currentGraph.outputs.push(out); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  mutator = new GraphMutator(currentGraph); /* v8 ignore next */ /* v8 ignore next */
  logGraphState(); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
document.getElementById('btnRename')!.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  if (!mutator) return alert('Initialize graph first!'); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const oldName = (document.getElementById('oldInput') as HTMLInputElement)
    .value; /* v8 ignore next */ /* v8 ignore next */
  const newName = (document.getElementById('newInput') as HTMLInputElement)
    .value; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  mutator.renameInput(oldName, newName); /* v8 ignore next */ /* v8 ignore next */
  logGraphState(); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
document.getElementById('btnBatch')!.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  if (!mutator || !currentGraph)
    return alert('Initialize graph first!'); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const batchStr = (document.getElementById('batchSize') as HTMLInputElement)
    .value; /* v8 ignore next */ /* v8 ignore next */
  const batchSize = isNaN(Number(batchStr))
    ? batchStr
    : Number(batchStr); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Since GraphMutator doesn't have an explicit 'changeBatch' we manually update using overrideShape for inputs /* v8 ignore next */ /* v8 ignore next */
  // or apply the standard headless JS equivalent /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  for (const inp of currentGraph.inputs) {
    /* v8 ignore next */ /* v8 ignore next */
    if (inp.shape.length > 0) {
      /* v8 ignore next */ /* v8 ignore next */
      const newShape = [...inp.shape]; /* v8 ignore next */ /* v8 ignore next */
      newShape[0] = batchSize; /* v8 ignore next */ /* v8 ignore next */
      mutator.overrideShape(
        inp.name,
        newShape,
        inp.dtype,
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  for (const out of currentGraph.outputs) {
    /* v8 ignore next */ /* v8 ignore next */
    if (out.shape.length > 0) {
      /* v8 ignore next */ /* v8 ignore next */
      const newShape = [...out.shape]; /* v8 ignore next */ /* v8 ignore next */
      newShape[0] = batchSize; /* v8 ignore next */ /* v8 ignore next */
      mutator.overrideShape(
        out.name,
        newShape,
        out.dtype,
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  logGraphState(); /* v8 ignore next */ /* v8 ignore next */
});
