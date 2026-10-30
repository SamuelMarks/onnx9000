/**
 * @fileoverview TF.js Shim Demo UI.
 */
import * as tf from '@onnx9000/tfjs-shim';

/**
 * Initializes the tfjs-shim demo UI.
 */
export function initTfjsShimDemo(): void {
  const runBtn = document.getElementById('run-btn');
  if (!runBtn) return;

  runBtn.addEventListener('click', () => {
    const out = document.getElementById('output');
    if (!out) return;

    out.innerText = `Running operations...

`;

    tf.tidy(() => {
      const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
      const b = tf.tensor2d([5, 6, 7, 8], [2, 2]);

      out.innerText += `Tensor A:
`;
      out.innerText += `Shape: ${a.shape}, Data: ${a.dataSync()}

`;

      out.innerText += `Tensor B:
`;
      out.innerText += `Shape: ${b.shape}, Data: ${b.dataSync()}

`;

      const c = tf.matMul(a, b);
      out.innerText += `C = matMul(A, B):
`;
      out.innerText += `Shape: ${c.shape}, Data: ${c.dataSync()}

`;

      const d = tf.relu(tf.sub(a, tf.scalar(2)));
      out.innerText += `D = relu(sub(A, 2)):
`;
      out.innerText += `Shape: ${d.shape}, Data: ${d.dataSync()}

`;
    });

    out.innerText += 'Operations completed inside tf.tidy scope.';
  });
}

initTfjsShimDemo();
