/**
 * @fileoverview Autograd Demo UI.
 */

/**
 * Initializes the Autograd demo UI.
 */
export function initAutogradDemo(): void {
  const gradBtn = document.getElementById('grad-btn') as HTMLButtonElement | null;
  const out = document.getElementById('autograd-output') as HTMLElement | null;

  if (!gradBtn || !out) return;

  gradBtn.addEventListener('click', async () => {
    out.innerText = `Building backward graph via AOTAutograd...
`;
    await new Promise((r) => setTimeout(r, 50));
    out.innerText += `Appending VJP nodes: MatMulGrad, AddGrad, ReluGrad...
Connecting Optimizer (AdamW) step nodes...
`;
    await new Promise((r) => setTimeout(r, 50));
    out.innerText += `
Success! Augmented ONNX graph now computes forward pass + gradients.`;
    gradBtn.disabled = true;
  });
}

initAutogradDemo();
