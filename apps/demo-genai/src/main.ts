const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
const out = document.getElementById('output') as HTMLElement;

runBtn.addEventListener('click', () => {
  out.innerText = 'Initializing GenAI Subsystem...';
  setTimeout(() => {
    out.innerText = 'GenAI models loaded.\nExecution complete: SUCCESS';
  }, 500);
});
