/**
 * @fileoverview ONNX Script Runner Demo UI.
 */
import { Graph } from '@onnx9000/core';

/**
 * Initializes the ONNX script runner demo UI.
 */
export function initOnnxScriptDemo(): void {
  const runBtn = document.getElementById('runBtn') as HTMLButtonElement | null;
  const outputDiv = document.getElementById('output') as HTMLDivElement | null;
  const scriptInput = document.getElementById('scriptInput') as HTMLTextAreaElement | null;

  if (!runBtn || !outputDiv || !scriptInput) return;

  runBtn.addEventListener('click', () => {
    outputDiv.textContent = 'Transpiling and running ONNX Script...';
    try {
      const scriptCode = scriptInput.value;
      const func = new Function('Graph', scriptCode);
      const result = func(Graph);

      outputDiv.textContent = `Success! Generated Graph JSON:

${JSON.stringify(result, null, 2)}`;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      outputDiv.textContent = `Error executing script: ${msg}`;
    }
  });
}

initOnnxScriptDemo();
