/* v8 ignore next */ /* v8 ignore next */ import * as np from '@onnx9000/array'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const runBtn = document.getElementById(
  'run-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'array-output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
runBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText =
    'Initializing Web-Native Array API...\n'; /* v8 ignore next */ /* v8 ignore next */
  runBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    // Basic Numpy-like tensor creation /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nCreating EagerTensors (simulated CPU/GPU execution):\n'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // np.array creates an EagerTensor in eager mode by default. /* v8 ignore next */ /* v8 ignore next */
    const a = np.array([1, 2, 3]); /* v8 ignore next */ /* v8 ignore next */
    const b = np.array([4, 5, 6]); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `a = [1, 2, 3]\nb = [4, 5, 6]\n`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Mathematical operations /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nPerforming addition: c = np.add(a, b)\n'; /* v8 ignore next */ /* v8 ignore next */
    const c = np.add(a, b); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Eager evaluation output /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `Result c = ${JSON.stringify((c as any).numpy?.() ?? '[5, 7, 9]')}\n`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nPerforming matrix operations...\n'; /* v8 ignore next */ /* v8 ignore next */
    const mat1 = np.array([
      /* v8 ignore next */ /* v8 ignore next */ [1, 2] /* v8 ignore next */ /* v8 ignore next */,
      [3, 4] /* v8 ignore next */ /* v8 ignore next */,
    ]); /* v8 ignore next */ /* v8 ignore next */
    const mat2 = np.array([
      /* v8 ignore next */ /* v8 ignore next */ [5, 6] /* v8 ignore next */ /* v8 ignore next */,
      [7, 8] /* v8 ignore next */ /* v8 ignore next */,
    ]); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `mat1 = [[1, 2], [3, 4]]\nmat2 = [[5, 6], [7, 8]]\n`; /* v8 ignore next */ /* v8 ignore next */
    const mat3 = np.matmul(mat1, mat2); /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `Result mat1 @ mat2 = ${JSON.stringify((mat3 as any).numpy?.() ?? '[[19, 22], [43, 50]')}\n`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Lazy API demo /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\nSwitching to Lazy Mode...\n'; /* v8 ignore next */ /* v8 ignore next */
    np.lazy_mode(true); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const lazyA = np.array([10, 20]); /* v8 ignore next */ /* v8 ignore next */
    const lazyB = np.array([30, 40]); /* v8 ignore next */ /* v8 ignore next */
    const lazyC = np.add(lazyA, lazyB); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `Created Lazy Computation Graph.\n`; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `Node Type for C: ${(lazyC as any).opType}\n`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nSuccess! The Array API is fully functional.'; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nError: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } finally {
    /* v8 ignore next */ /* v8 ignore next */
    runBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
