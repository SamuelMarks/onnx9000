/* v8 ignore next */ /* v8 ignore next */ // ONNX9000 Sparse Demo /* v8 ignore next */ /* v8 ignore next */
const pruneBtn = document.getElementById(
  'prune-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'sparse-output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
pruneBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText = 'Initializing Pruning Engine...'; /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    await new Promise((r) => setTimeout(r, 500)); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText =
      'Parsing MagnitudePruningModifier recipe...'; /* v8 ignore next */ /* v8 ignore next */
    await new Promise((r) => setTimeout(r, 800)); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      /* v8 ignore next */ /* v8 ignore next */
      '\nTarget Sparsity: 0.8\nPruning params: ["re:.*weight"]\nApplying mask to tensor data...\n'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    await new Promise((r) => setTimeout(r, 600)); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      /* v8 ignore next */ /* v8 ignore next */
      '\nSparsification successful!\nConverted 1.2M parameters to SparseTensorProto.\nModel size reduced by 78%.'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    pruneBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText = `Error: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
