import { Graph } from '@onnx9000/core';

document.addEventListener('DOMContentLoaded', () => {
  const runBtn = document.getElementById('runBtn') as HTMLButtonElement;
  const scriptInput = document.getElementById('scriptInput') as HTMLTextAreaElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;

  runBtn.addEventListener('click', () => {
    outputDiv.textContent = 'Running...';
    try {
      // Evaluate the user script. We wrap it in a function providing the Graph API.
      const scriptCode = scriptInput.value;

      // CAUTION: new Function is used here for demonstration of fluent scripting evaluation in a demo context.
      // In production, user input should be properly sanitized or run in an isolated environment.
      const func = new Function('Graph', scriptCode);
      const result = func(Graph);

      outputDiv.textContent =
        'Success! Generated Graph JSON:\n\n' + JSON.stringify(result, null, 2);
    } catch (err: any) {
      outputDiv.textContent = 'Error executing script: ' + err.message;
    }
  });
});
