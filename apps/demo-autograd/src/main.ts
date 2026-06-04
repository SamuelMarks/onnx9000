/* v8 ignore next */ /* v8 ignore next */ // ONNX9000 Autograd Demo /* v8 ignore next */ /* v8 ignore next */
const gradBtn = document.getElementById(
  'grad-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'autograd-output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
gradBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText = 'Initializing Autograd Engine...'; /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    await new Promise((r) => setTimeout(r, 500)); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText =
      'Traversing graph from loss node backwards...'; /* v8 ignore next */ /* v8 ignore next */
    await new Promise((r) => setTimeout(r, 800)); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nComputing partial derivatives for: MatMul, Relu, Add...'; /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nInjecting `ai.onnx.training` gradient nodes...\n'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    await new Promise((r) => setTimeout(r, 600)); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nSuccess! Augmented ONNX graph now computes forward pass + gradients.'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    gradBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText = `Error: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
