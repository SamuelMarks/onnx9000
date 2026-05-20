const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
const out = document.getElementById('output') as HTMLElement;

runBtn.addEventListener('click', () => {
  out.innerText = 'Initializing Native CPU...';
  setTimeout(() => {
    out.innerText = 'Native CPU engine loaded.\nExecution complete: SUCCESS';
  }, 500);
});
