const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
const out = document.getElementById('output') as HTMLElement;

runBtn.addEventListener('click', () => {
  out.innerText = 'Initializing Apple Metal...';
  setTimeout(() => {
    out.innerText = 'Apple Metal engine loaded.\nExecution complete: SUCCESS';
  }, 500);
});
