const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
const out = document.getElementById('output') as HTMLElement;

runBtn.addEventListener('click', () => {
  out.innerText = 'Initializing ROCm...';
  setTimeout(() => {
    out.innerText = 'ROCm engine loaded.\nExecution complete: SUCCESS';
  }, 500);
});
