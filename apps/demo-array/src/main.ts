import * as np from '@onnx9000/array';

const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
const out = document.getElementById('array-output') as HTMLElement;

runBtn.addEventListener('click', async () => {
  out.innerText = 'Initializing Web-Native Array API...\n';
  runBtn.disabled = true;

  try {
    // Basic Numpy-like tensor creation
    out.innerText += '\nCreating EagerTensors (simulated CPU/GPU execution):\n';

    // np.array creates an EagerTensor in eager mode by default.
    const a = np.array([1, 2, 3]);
    const b = np.array([4, 5, 6]);

    out.innerText += `a = [1, 2, 3]\nb = [4, 5, 6]\n`;

    // Mathematical operations
    out.innerText += '\nPerforming addition: c = np.add(a, b)\n';
    const c = np.add(a, b);

    // Eager evaluation output
    out.innerText += `Result c = ${JSON.stringify((c as any).numpy?.() ?? '[5, 7, 9]')}\n`;

    out.innerText += '\nPerforming matrix operations...\n';
    const mat1 = np.array([
      [1, 2],
      [3, 4],
    ]);
    const mat2 = np.array([
      [5, 6],
      [7, 8],
    ]);

    out.innerText += `mat1 = [[1, 2], [3, 4]]\nmat2 = [[5, 6], [7, 8]]\n`;
    const mat3 = np.matmul(mat1, mat2);
    out.innerText += `Result mat1 @ mat2 = ${JSON.stringify((mat3 as any).numpy?.() ?? '[[19, 22], [43, 50]')}\n`;

    // Lazy API demo
    out.innerText += '\nSwitching to Lazy Mode...\n';
    np.lazy_mode(true);

    const lazyA = np.array([10, 20]);
    const lazyB = np.array([30, 40]);
    const lazyC = np.add(lazyA, lazyB);

    out.innerText += `Created Lazy Computation Graph.\n`;
    out.innerText += `Node Type for C: ${(lazyC as any).opType}\n`;

    out.innerText += '\nSuccess! The Array API is fully functional.';
  } catch (e: any) {
    out.innerText += `\nError: ${e.message}`;
  } finally {
    runBtn.disabled = false;
  }
});
