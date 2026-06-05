document.getElementById('btn-compile')?.addEventListener('click', () => {
  const output = document.getElementById('output');
  if (output) {
    output.textContent = 'Compiling...\n';
    setTimeout(() => {
      output.textContent += '[OK] Read model.onnx\n';
      output.textContent += '[OK] Lowering to generic IR...\n';
      output.textContent += '[OK] Applying optimizations...\n';
      output.textContent += '[OK] AOT Compilation finished: model.bin';
    }, 500);
  }
});
