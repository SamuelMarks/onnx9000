import * as tf from '@onnx9000/tfjs-shim';

document.getElementById('run-btn')!.addEventListener('click', async () => {
  const out = document.getElementById('output')!;
  out.innerText = 'Running operations...\n\n';

  // Demonstrate basic tensor creation and operations
  tf.tidy(() => {
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
    const b = tf.tensor2d([5, 6, 7, 8], [2, 2]);

    out.innerText += 'Tensor A:\n';
    out.innerText += `Shape: ${a.shape}, Data: ${a.dataSync()}\n\n`;

    out.innerText += 'Tensor B:\n';
    out.innerText += `Shape: ${b.shape}, Data: ${b.dataSync()}\n\n`;

    const c = tf.matMul(a, b);
    out.innerText += 'C = matMul(A, B):\n';
    out.innerText += `Shape: ${c.shape}, Data: ${c.dataSync()}\n\n`;

    const d = tf.relu(tf.sub(a, tf.scalar(2)));
    out.innerText += 'D = relu(sub(A, 2)):\n';
    out.innerText += `Shape: ${d.shape}, Data: ${d.dataSync()}\n\n`;
  });

  out.innerText += 'Operations completed inside tf.tidy scope.';
});
