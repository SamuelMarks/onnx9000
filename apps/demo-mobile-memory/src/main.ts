document.addEventListener('DOMContentLoaded', () => {
  const allocateBtn = document.getElementById('allocateBtn') as HTMLButtonElement;
  const runInferenceBtn = document.getElementById('runInferenceBtn') as HTMLButtonElement;
  const freeBtn = document.getElementById('freeBtn') as HTMLButtonElement;
  const arenaContainer = document.getElementById('arena-container') as HTMLDivElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;

  let arena: ArrayBuffer | null = null;
  const NUM_BLOCKS = 20;

  const log = (msg: string) => {
    outputDiv.textContent += `\n${msg}`;
    outputDiv.scrollTop = outputDiv.scrollHeight;
  };

  const renderArena = (state: 'empty' | 'allocated' | 'in-use') => {
    arenaContainer.innerHTML = '';
    if (state === 'empty') return;

    for (let i = 0; i < NUM_BLOCKS; i++) {
      const block = document.createElement('div');
      block.className = 'memory-block';
      if (state === 'in-use' && Math.random() > 0.3) {
        block.classList.add('allocated');
        block.textContent = 'busy';
      } else {
        block.textContent = 'free';
      }
      arenaContainer.appendChild(block);
    }
  };

  allocateBtn.addEventListener('click', () => {
    outputDiv.textContent = 'Pre-allocating 10MB contiguous ArrayBuffer...';
    try {
      arena = new ArrayBuffer(10 * 1024 * 1024);
      renderArena('allocated');
      log('Arena pre-allocated successfully. Dynamic allocations eliminated.');

      allocateBtn.disabled = true;
      runInferenceBtn.disabled = false;
      freeBtn.disabled = false;
    } catch (err: any) {
      log('Failed to allocate: ' + err.message);
    }
  });

  runInferenceBtn.addEventListener('click', () => {
    if (!arena) return;
    log('Running inference pass using static memory arena...');
    renderArena('in-use');

    // Simulate inference work
    setTimeout(() => {
      renderArena('allocated');
      log('Inference complete. No memory was allocated or garbage collected.');
    }, 500);
  });

  freeBtn.addEventListener('click', () => {
    arena = null;
    renderArena('empty');
    log('Arena memory freed.');
    allocateBtn.disabled = false;
    runInferenceBtn.disabled = true;
    freeBtn.disabled = true;
  });
});
