const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
const out = document.getElementById('output') as HTMLElement;

runBtn.addEventListener('click', () => {
  out.innerText = 'Initializing WASM Compiler...';
  setTimeout(() => {
    out.innerText = 'WASM engine loaded.\nExecution complete: SUCCESS';
  }, 500);
});
