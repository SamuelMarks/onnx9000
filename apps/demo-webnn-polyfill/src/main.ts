// We import the polyfill to ensure navigator.ml is injected
import '@onnx9000/webnn-polyfill';

const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
const out = document.getElementById('webnn-output') as HTMLElement;

runBtn.addEventListener('click', async () => {
  out.innerText = 'Initializing WebNN Context...\n';
  runBtn.disabled = true;

  try {
    const ml = (navigator as any).ml;
    if (!ml) {
      throw new Error('WebNN polyfill failed to inject navigator.ml');
    }

    const context = await ml.createContext({ deviceType: 'gpu' });
    out.innerText += '\nCreated MLContext.';

    // WebNN API is exposed globally by the polyfill
    const builder = new (window as any).MLGraphBuilder(context);

    // Build a simple y = W * x + b graph
    out.innerText += '\nBuilding computational graph...';

    const xDesc = { dataType: 'float32', dimensions: [1, 2] };
    const wDesc = { dataType: 'float32', dimensions: [2, 2] };
    const bDesc = { dataType: 'float32', dimensions: [2] };

    const x = builder.input('x', xDesc);
    const w = builder.constant(wDesc, new Float32Array([1, 2, 3, 4]));
    const b = builder.constant(bDesc, new Float32Array([0.5, 0.5]));

    // Matmul
    const matmul = builder.matmul(x, w);

    // Add
    const y = builder.add(matmul, b);

    // Compile
    out.innerText += '\nCompiling graph...';
    const graph = await builder.build({ y });

    // Execute
    out.innerText += '\nExecuting graph with input x = [1, 1]...';
    const inputs = { x: new Float32Array([1, 1]) };
    const outputs = { y: new Float32Array(2) };

    const results = await context.compute(graph, inputs, outputs);

    out.innerText += `\n\nResult y: [${results.outputs.y.join(', ')}]`;
    out.innerText += '\n\nSuccess! WebNN API execution complete.';
  } catch (e: any) {
    out.innerText += `\nError: ${e.message}`;
  } finally {
    runBtn.disabled = false;
  }
});
