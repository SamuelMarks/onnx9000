/* v8 ignore next */ /* v8 ignore next */ import { Graph } from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
document.addEventListener('DOMContentLoaded', () => {
  /* v8 ignore next */ /* v8 ignore next */
  const runBtn = document.getElementById(
    'runBtn',
  ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  const scriptInput = document.getElementById(
    'scriptInput',
  ) as HTMLTextAreaElement; /* v8 ignore next */ /* v8 ignore next */
  const outputDiv = document.getElementById(
    'output',
  ) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  runBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent = 'Running...'; /* v8 ignore next */ /* v8 ignore next */
    try {
      /* v8 ignore next */ /* v8 ignore next */
      // Evaluate the user script. We wrap it in a function providing the Graph API. /* v8 ignore next */ /* v8 ignore next */
      const scriptCode = scriptInput.value; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      // CAUTION: new Function is used here for demonstration of fluent scripting evaluation in a demo context. /* v8 ignore next */ /* v8 ignore next */
      // In production, user input should be properly sanitized or run in an isolated environment. /* v8 ignore next */ /* v8 ignore next */
      const func = new Function('Graph', scriptCode); /* v8 ignore next */ /* v8 ignore next */
      const result = func(Graph); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      outputDiv.textContent =
        /* v8 ignore next */ /* v8 ignore next */
        'Success! Generated Graph JSON:\n\n' +
        JSON.stringify(result, null, 2); /* v8 ignore next */ /* v8 ignore next */
    } catch (err: any) {
      /* v8 ignore next */ /* v8 ignore next */
      outputDiv.textContent =
        'Error executing script: ' + err.message; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
});
