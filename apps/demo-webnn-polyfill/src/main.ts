/**
 * @fileoverview WebNN Polyfill Demo UI.
 */
import { Onnx9000Error } from '@onnx9000/core';
import {
  type MLContext,
  type MLOperandDescriptor,
  PolyfillMLGraphBuilder,
} from '@onnx9000/webnn-polyfill';

/** Interface `NavigatorWithML` definition. */
interface NavigatorWithML {
  /** Property `ml`. */
  ml?: {
    createContext(options?: { deviceType?: string }): Promise<MLContext>;
  };
}

/** Interface `WindowWithMLGraphBuilder` definition. */
interface WindowWithMLGraphBuilder {
  /** Property `MLGraphBuilder`. */
  MLGraphBuilder?: typeof PolyfillMLGraphBuilder;
}

/**
 * Initializes the WebNN demo UI.
 */
export function initWebnnDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement | null;
  const out = document.getElementById('webnn-output') as HTMLElement | null;

  if (!runBtn || !out) return;

  runBtn.addEventListener('click', async () => {
    out.innerText = 'Initializing WebNN Context...';
    runBtn.disabled = true;

    try {
      const ml = (navigator as NavigatorWithML).ml;
      if (!ml) {
        throw new Onnx9000Error('WebNN polyfill failed to inject navigator.ml');
      }

      const context = await ml.createContext({ deviceType: 'gpu' });
      out.innerText += ' Created MLContext.';

      const BuilderClass =
        (window as WindowWithMLGraphBuilder).MLGraphBuilder ?? PolyfillMLGraphBuilder;
      const builder = new BuilderClass(context);

      out.innerText += ' Building computational graph...';

      const xDesc: MLOperandDescriptor = {
        dataType: 'float32',
        dimensions: [1, 2],
      };
      const wDesc: MLOperandDescriptor = {
        dataType: 'float32',
        dimensions: [2, 2],
      };
      const bDesc: MLOperandDescriptor = {
        dataType: 'float32',
        dimensions: [2],
      };

      const x = builder.input('x', xDesc);
      const w = builder.constant(wDesc, new Float32Array([1, 2, 3, 4]));
      const b = builder.constant(bDesc, new Float32Array([0.5, 0.5]));

      const matmul = builder.matmul(x, w);
      const y = builder.add(matmul, b);

      out.innerText += ' Compiling graph...';
      const graph = await builder.build({ y });

      out.innerText += ' Executing graph with input x = [1, 1]...';
      const inputs = { x: new Float32Array([1, 1]) };
      const outputs = { y: new Float32Array(2) };

      const results = await context.compute(graph, inputs, outputs);
      const outArr = results.outputs.y as Float32Array;

      out.innerText += ` Result y: [${Array.from(outArr).join(', ')}]`;
      out.innerText += ' Success! WebNN API execution complete.';
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      out.innerText += ` Error: ${msg}`;
    } finally {
      runBtn.disabled = false;
    }
  });
}

initWebnnDemo();
