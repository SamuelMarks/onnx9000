/* v8 ignore start */
import { Graph } from '@onnx9000/core';

/**
 * Initializes the ONNX script runner demo UI.
 */
export function initOnnxScriptDemo(): void {
  const runBtn = document.getElementById('runBtn') as HTMLButtonElement;
  const scriptInput = document.getElementById('scriptInput') as HTMLTextAreaElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;

  if (!runBtn || !scriptInput || !outputDiv) return;

  runBtn.addEventListener('click', () => {
    outputDiv.textContent = 'Running...';
    try {
      const scriptCode = scriptInput.value;
      const func = new Function('Graph', scriptCode);
      const result = func(Graph);

      outputDiv.textContent =
        'Success! Generated Graph JSON:\n\n' + JSON.stringify(result, null, 2);
    } catch (err: any) {
      outputDiv.textContent = 'Error executing script: ' + err.message;
    }
  });
}

document.addEventListener('DOMContentLoaded', initOnnxScriptDemo);
if (document.readyState === 'complete' || document.readyState === 'interactive') {
  initOnnxScriptDemo();
}

/* v8 ignore stop */
