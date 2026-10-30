/**
 * @fileoverview Mobile Memory Arena Demo UI.
 */

/**
 * Initializes the mobile memory arena demo.
 */
export function initMobileMemoryDemo(): void {
  const allocateBtn = document.getElementById('allocateBtn') as HTMLButtonElement | null;
  const runInferenceBtn = document.getElementById('runInferenceBtn') as HTMLButtonElement | null;
  const freeBtn = document.getElementById('freeBtn') as HTMLButtonElement | null;
  const arenaContainer = document.getElementById('arena-container') as HTMLDivElement | null;
  const outputDiv = document.getElementById('output') as HTMLDivElement | null;

  if (!allocateBtn || !runInferenceBtn || !freeBtn || !arenaContainer || !outputDiv) return;

  let arena: ArrayBuffer | null = null;
  const NUM_BLOCKS = 20;

  const log = (msg: string) => {
    outputDiv.textContent += `
${msg}`;
    outputDiv.scrollTop = outputDiv.scrollHeight;
  };

  const renderArena = (state: 'empty' | 'allocated' | 'in-use') => {
    arenaContainer.innerHTML = '';
    if (state === 'empty') return;

    for (let i = 0; i < NUM_BLOCKS; i++) {
      const block = document.createElement('div');
      block.className = 'memory-block';
      if (state === 'in-use' && i % 2 === 0) {
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
    arena = new ArrayBuffer(10 * 1024 * 1024);
    renderArena('allocated');
    log('Arena pre-allocated successfully. Dynamic allocations eliminated.');

    allocateBtn.disabled = true;
    runInferenceBtn.disabled = false;
    freeBtn.disabled = false;
  });

  runInferenceBtn.addEventListener('click', () => {
    if (!arena) return;
    log('Running inference pass using static memory arena...');
    renderArena('in-use');

    setTimeout(() => {
      renderArena('allocated');
      log('Inference complete. No memory was allocated or garbage collected.');
    }, 40);
  });

  freeBtn.addEventListener('click', () => {
    arena = null;
    renderArena('empty');
    log('Arena memory freed.');
    allocateBtn.disabled = false;
    runInferenceBtn.disabled = true;
    freeBtn.disabled = true;
  });
}

initMobileMemoryDemo();
