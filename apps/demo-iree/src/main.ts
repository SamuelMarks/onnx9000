/* v8 ignore start */
import { compileModel } from "@onnx9000/iree-compiler/src/cli.js";
import {
  Module,
  Context,
  WVMInterpreter,
  HALBindings,
} from "@onnx9000/iree-runtime/src/vm.js";

/**
 * Initializes the IREE demo UI.
 */
export function initIreeDemo(): void {
  const compileBtn = document.getElementById(
    "compile-btn",
  ) as HTMLButtonElement;
  const runBtn = document.getElementById("run-btn") as HTMLButtonElement;
  const compileOut = document.getElementById("compiler-output") as HTMLElement;
  const runOut = document.getElementById("runtime-output") as HTMLElement;

  if (!compileBtn || !runBtn || !compileOut || !runOut) return;

  let bytecode: Uint8Array | null = null;

  compileBtn.addEventListener("click", async () => {
    compileOut.innerText = "Compiling...";
    try {
      await compileModel("dummy.onnx", {
        targetBackend: "wasm",
        dumpMlir: true,
        optimizeLevel: "O3",
      });

      compileOut.innerText =
        "Compilation successful.\nGenerated simulated WVM bytecode.\nBackend: wasm\nAggressive O3 optims applied.";
      bytecode = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x03, 0xff]);
      runBtn.disabled = false;
    } catch (e: any) {
      compileOut.innerText = `Compiler Error: ${e.message}`;
    }
  });

  runBtn.addEventListener("click", async () => {
    if (!bytecode) return;
    runOut.innerText = "Initializing Web VM...";

    try {
      const mod = new Module();
      const ctx = new Context(mod);
      HALBindings.register(ctx, { mockDevice: true });
      const interpreter = new WVMInterpreter(bytecode, ctx);

      const start = performance.now();
      await interpreter.runAsync();
      const duration = performance.now() - start;

      runOut.innerText = `Execution successful!\nTime: ${duration.toFixed(2)}ms\nInstructions processed perfectly.`;
    } catch (e: any) {
      runOut.innerText = `Runtime Error: ${e.message}`;
    }
  });
}
initIreeDemo();

/* v8 ignore stop */
