/**
 * @fileoverview AOT/JIT Compile Demo UI.
 */

/**
 * Initializes the compile demo UI.
 */
export function initCompileDemo(): void {
  document.getElementById('btn-compile')?.addEventListener('click', () => {
    const output = document.getElementById('output');
    if (output) {
      output.textContent = `Compiling...
`;
      setTimeout(() => {
        output.textContent += `[OK] Read model.onnx
`;
        output.textContent += `[OK] Lowering to generic IR...
`;
        output.textContent += `[OK] Applying optimizations...
`;
        output.textContent += '[OK] AOT Compilation finished: model.bin';
      }, 50);
    }
  });
}

initCompileDemo();
