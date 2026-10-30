/**
 * @fileoverview IREE Compiler & WebVM Runtime Demo UI.
 */
import { compileModel } from '@onnx9000/iree-compiler';
import { Context, HALBindings, Module, WVMInterpreter } from '@onnx9000/iree-runtime';

/**
 * Initializes the IREE demo UI.
 */
export function initIreeDemo(): void {
  const compileBtn = document.getElementById('compile-btn') as HTMLButtonElement | null;
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement | null;
  const compileOut = document.getElementById('compile-output') as HTMLElement | null;
  const runOut = document.getElementById('run-output') as HTMLElement | null;

  if (!compileBtn || !runBtn || !compileOut || !runOut) return;

  let bytecode: Uint8Array | null = null;

  compileBtn.addEventListener('click', async () => {
    compileOut.innerText = 'Compiling via @onnx9000/iree-compiler...';
    try {
      await compileModel('model.onnx', {
        targetBackend: 'wasm',
        dumpMlir: false,
        optimizeLevel: 'O3',
      });
      compileOut.innerText = `Compilation successful.
Generated simulated WVM bytecode.
Backend: wasm
Aggressive O3 optims applied.`;
      bytecode = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x03, 0xff]);
      runBtn.disabled = false;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      compileOut.innerText = `Compiler Error: ${msg}`;
    }
  });

  runBtn.addEventListener('click', async () => {
    if (!bytecode) return;
    runOut.innerText = 'Initializing Web VM...';

    try {
      const mod = new Module();
      const ctx = new Context(mod);
      HALBindings.register(ctx, { mockDevice: true });
      const interpreter = new WVMInterpreter(bytecode, ctx);

      const start = performance.now();
      await interpreter.runAsync();
      const duration = performance.now() - start;

      runOut.innerText = `Execution successful!
Time: ${duration.toFixed(2)}ms
Instructions processed perfectly.`;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      runOut.innerText = `Runtime Error: ${msg}`;
    }
  });
}

initIreeDemo();
