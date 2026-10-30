/**
 * @fileoverview Sparse Pruning Demo UI.
 */

/**
 * Initializes the sparse demo UI.
 */
export function initSparseDemo(): void {
  const pruneBtn = document.getElementById('prune-btn') as HTMLButtonElement | null;
  const out = document.getElementById('sparse-output') as HTMLElement | null;
  if (!pruneBtn || !out) return;

  pruneBtn.addEventListener('click', async () => {
    out.innerText = `Reading SparseML Recipe...
`;
    await new Promise((r) => setTimeout(r, 50));
    out.innerText += `
Target Sparsity: 0.8
Pruning params: ["re:.*weight"]
Applying mask to tensor data...
`;
    await new Promise((r) => setTimeout(r, 50));
    out.innerText += `
Sparsification successful!
Converted 1.2M parameters to SparseTensorProto.
Model size reduced by 78%.`;
    pruneBtn.disabled = true;
  });
}

initSparseDemo();
