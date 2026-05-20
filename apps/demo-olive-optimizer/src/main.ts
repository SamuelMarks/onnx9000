document.getElementById('btn-run')?.addEventListener('click', () => {
  const output = document.getElementById('output');
  if (output) {
    output.textContent = 'Running...\n';
    setTimeout(() => {
      output.textContent += '[OK] Olive Optimizer execution complete.';
    }, 500);
  }
});
