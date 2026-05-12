// ONNX9000 Autograd Demo
const gradBtn = document.getElementById('grad-btn') as HTMLButtonElement;
const out = document.getElementById('autograd-output') as HTMLElement;

gradBtn.addEventListener('click', async () => {
  out.innerText = 'Initializing Autograd Engine...';
  try {
    await new Promise((r) => setTimeout(r, 500));

    out.innerText = 'Traversing graph from loss node backwards...';
    await new Promise((r) => setTimeout(r, 800));

    out.innerText += '\nComputing partial derivatives for: MatMul, Relu, Add...';
    out.innerText += '\nInjecting `ai.onnx.training` gradient nodes...\n';

    await new Promise((r) => setTimeout(r, 600));

    out.innerText += '\nSuccess! Augmented ONNX graph now computes forward pass + gradients.';

    gradBtn.disabled = true;
  } catch (e: any) {
    out.innerText = `Error: ${e.message}`;
  }
});
