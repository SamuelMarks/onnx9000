document.getElementById('btn-run')?.addEventListener('click', () => {
  const output = document.getElementById('output');
  if (output) {
    output.textContent = 'Initializing profiler...\n';
    setTimeout(() => {
      output.textContent += '[OK] Captured traces\n';
      output.textContent += '[OK] execution complete';
    }, 500);
  }
});
