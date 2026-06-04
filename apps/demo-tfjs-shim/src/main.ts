/* v8 ignore next */ /* v8 ignore next */ import * as tf from '@onnx9000/tfjs-shim'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
document.getElementById('run-btn')!.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  const out = document.getElementById('output')!; /* v8 ignore next */ /* v8 ignore next */
  out.innerText = 'Running operations...\n\n'; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Demonstrate basic tensor creation and operations /* v8 ignore next */ /* v8 ignore next */
  tf.tidy(() => {
    /* v8 ignore next */ /* v8 ignore next */
    const a = tf.tensor2d([1, 2, 3, 4], [2, 2]); /* v8 ignore next */ /* v8 ignore next */
    const b = tf.tensor2d([5, 6, 7, 8], [2, 2]); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += 'Tensor A:\n'; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `Shape: ${a.shape}, Data: ${a.dataSync()}\n\n`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += 'Tensor B:\n'; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `Shape: ${b.shape}, Data: ${b.dataSync()}\n\n`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const c = tf.matMul(a, b); /* v8 ignore next */ /* v8 ignore next */
    out.innerText += 'C = matMul(A, B):\n'; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `Shape: ${c.shape}, Data: ${c.dataSync()}\n\n`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const d = tf.relu(tf.sub(a, tf.scalar(2))); /* v8 ignore next */ /* v8 ignore next */
    out.innerText += 'D = relu(sub(A, 2)):\n'; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `Shape: ${d.shape}, Data: ${d.dataSync()}\n\n`; /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText +=
    'Operations completed inside tf.tidy scope.'; /* v8 ignore next */ /* v8 ignore next */
});
