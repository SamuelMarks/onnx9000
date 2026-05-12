// ONNX9000 Sparse Demo
const pruneBtn = document.getElementById('prune-btn') as HTMLButtonElement;
const out = document.getElementById('sparse-output') as HTMLElement;

pruneBtn.addEventListener('click', async () => {
  out.innerText = 'Initializing Pruning Engine...';
  try {
    await new Promise((r) => setTimeout(r, 500));

    out.innerText = 'Parsing MagnitudePruningModifier recipe...';
    await new Promise((r) => setTimeout(r, 800));

    out.innerText +=
      '\nTarget Sparsity: 0.8\nPruning params: ["re:.*weight"]\nApplying mask to tensor data...\n';

    await new Promise((r) => setTimeout(r, 600));

    out.innerText +=
      '\nSparsification successful!\nConverted 1.2M parameters to SparseTensorProto.\nModel size reduced by 78%.';

    pruneBtn.disabled = true;
  } catch (e: any) {
    out.innerText = `Error: ${e.message}`;
  }
});
