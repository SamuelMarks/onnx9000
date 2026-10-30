/**
 * @fileoverview Hummingbird Transpiler Demo UI.
 */

/**
 * Initializes the Hummingbird transpiler demo.
 */
export function initHummingbirdDemo(): void {
  const transpileBtn = document.getElementById('transpile-btn') as HTMLButtonElement | null;
  const out = document.getElementById('hb-output') as HTMLElement | null;
  if (!transpileBtn || !out) return;

  transpileBtn.addEventListener('click', async () => {
    out.innerText = `Parsing TreeEnsembleClassifier...
`;
    await new Promise((r) => setTimeout(r, 50));
    out.innerText += `Applying GEMM Strategy...
Mapping decision trees to MatMul and Less/Greater operations...
`;
    await new Promise((r) => setTimeout(r, 50));
    out.innerText += `
Transpilation successful!
Generated standard ONNX Tensor graph for WebGPU acceleration.`;
    transpileBtn.disabled = true;
  });
}

initHummingbirdDemo();
