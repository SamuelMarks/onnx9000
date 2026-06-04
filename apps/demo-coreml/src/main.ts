/* v8 ignore next */ /* v8 ignore next */ import {
  Graph,
  Node,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import { convertToCoreML } from '@onnx9000/coreml'; /* v8 ignore next */ /* v8 ignore next */
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
  // Build a mock ONNX graph /* v8 ignore next */ /* v8 ignore next */
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
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const milAst = convertToCoreML(g); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Custom replacer to handle bigint /* v8 ignore next */ /* v8 ignore next */
    const jsonString = JSON.stringify(
      /* v8 ignore next */ /* v8 ignore next */
      milAst /* v8 ignore next */ /* v8 ignore next */,
      (key, value) => {
        /* v8 ignore next */ /* v8 ignore next */
        if (typeof value === 'bigint') {
          /* v8 ignore next */ /* v8 ignore next */
          return value.toString() + 'n'; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        return value; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */,
      2 /* v8 ignore next */ /* v8 ignore next */,
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText = jsonString; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText = `Error: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
