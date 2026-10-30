/**
 * @fileoverview Array API Demo UI.
 */
import * as np from '@onnx9000/array';

/**
 * Initializes the Array API demo UI.
 */
export function initArrayDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement | null;
  const out = document.getElementById('array-output') as HTMLElement | null;

  if (!runBtn || !out) return;

  runBtn.addEventListener('click', () => {
    out.innerText = `Initializing Array Engine...
`;
    runBtn.disabled = true;

    const a = np.array([1, 2, 3]);
    const b = np.array([4, 5, 6]);

    out.innerText += `Created Tensor a = [1, 2, 3], b = [4, 5, 6]
`;
    out.innerText += `
Performing addition: c = np.add(a, b)
`;
    const c = np.add(a, b);
    out.innerText += `Result c = ${JSON.stringify(c.numpy())}
`;

    out.innerText += `
Performing matrix operations...
`;
    const mat1 = np.array([
      [1, 2],
      [3, 4],
    ]);
    const mat2 = np.array([
      [5, 6],
      [7, 8],
    ]);

    out.innerText += `mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6], [7, 8]]
`;
    const mat3 = np.matmul(mat1, mat2);
    out.innerText += `Result mat1 @ mat2 = ${JSON.stringify(mat3.numpy())}
`;

    out.innerText += `
Switching to Lazy Mode...
`;
    np.lazy_mode(true);

    const lazyA = np.array([10, 20]);
    const lazyB = np.array([30, 40]);
    const lazyC = np.add(lazyA, lazyB) as np.LazyTensor;

    out.innerText += `Created Lazy Computation Graph.
`;
    out.innerText += `Node Type for C: ${lazyC.opType}
`;
    np.lazy_mode(false);

    out.innerText += `
Success! The Array API is fully functional.`;
    runBtn.disabled = false;
  });
}

initArrayDemo();
