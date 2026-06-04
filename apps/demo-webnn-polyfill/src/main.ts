/* v8 ignore next */ /* v8 ignore next */ // We import the polyfill to ensure navigator.ml is injected /* v8 ignore next */ /* v8 ignore next */
import '@onnx9000/webnn-polyfill'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const runBtn = document.getElementById(
  'run-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'webnn-output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
runBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText = 'Initializing WebNN Context...\n'; /* v8 ignore next */ /* v8 ignore next */
  runBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const ml = (navigator as any).ml; /* v8 ignore next */ /* v8 ignore next */
    if (!ml) {
      /* v8 ignore next */ /* v8 ignore next */
      throw new Error(
        'WebNN polyfill failed to inject navigator.ml',
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const context = await ml.createContext({
      deviceType: 'gpu',
    }); /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\nCreated MLContext.'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // WebNN API is exposed globally by the polyfill /* v8 ignore next */ /* v8 ignore next */
    const builder = new (window as any).MLGraphBuilder(
      context,
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Build a simple y = W * x + b graph /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\nBuilding computational graph...'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const xDesc = {
      dataType: 'float32',
      dimensions: [1, 2],
    }; /* v8 ignore next */ /* v8 ignore next */
    const wDesc = {
      dataType: 'float32',
      dimensions: [2, 2],
    }; /* v8 ignore next */ /* v8 ignore next */
    const bDesc = {
      dataType: 'float32',
      dimensions: [2],
    }; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const x = builder.input('x', xDesc); /* v8 ignore next */ /* v8 ignore next */
    const w = builder.constant(
      wDesc,
      new Float32Array([1, 2, 3, 4]),
    ); /* v8 ignore next */ /* v8 ignore next */
    const b = builder.constant(
      bDesc,
      new Float32Array([0.5, 0.5]),
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Matmul /* v8 ignore next */ /* v8 ignore next */
    const matmul = builder.matmul(x, w); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Add /* v8 ignore next */ /* v8 ignore next */
    const y = builder.add(matmul, b); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Compile /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\nCompiling graph...'; /* v8 ignore next */ /* v8 ignore next */
    const graph = await builder.build({ y }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Execute /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nExecuting graph with input x = [1, 1]...'; /* v8 ignore next */ /* v8 ignore next */
    const inputs = { x: new Float32Array([1, 1]) }; /* v8 ignore next */ /* v8 ignore next */
    const outputs = { y: new Float32Array(2) }; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const results = await context.compute(
      graph,
      inputs,
      outputs,
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\n\nResult y: [${results.outputs.y.join(', ')}]`; /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\n\nSuccess! WebNN API execution complete.'; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nError: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } finally {
    /* v8 ignore next */ /* v8 ignore next */
    runBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
