document.getElementById('btn-convert')?.addEventListener('click', () => {
  const output = document.getElementById('output');
  if (output) {
    output.textContent = 'Parsing Scikit-LearnScikit-Learn structure...\n';
    setTimeout(() => {
      output.textContent += '[OK] Transpiled ops to ONNX nodes\n';
      output.textContent += '[OK] SKL2ONNX conversion complete.';
    }, 500);
  }
});
