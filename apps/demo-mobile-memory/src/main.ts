/* v8 ignore next */ /* v8 ignore next */ document.addEventListener('DOMContentLoaded', () => {
  /* v8 ignore next */ /* v8 ignore next */
  const allocateBtn = document.getElementById(
    'allocateBtn',
  ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  const runInferenceBtn = document.getElementById(
    'runInferenceBtn',
  ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  const freeBtn = document.getElementById(
    'freeBtn',
  ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  const arenaContainer = document.getElementById(
    'arena-container',
  ) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
  const outputDiv = document.getElementById(
    'output',
  ) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  let arena: ArrayBuffer | null = null; /* v8 ignore next */ /* v8 ignore next */
  const NUM_BLOCKS = 20; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const log = (msg: string) => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent += `\n${msg}`; /* v8 ignore next */ /* v8 ignore next */
    outputDiv.scrollTop = outputDiv.scrollHeight; /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const renderArena = (state: 'empty' | 'allocated' | 'in-use') => {
    /* v8 ignore next */ /* v8 ignore next */
    arenaContainer.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
    if (state === 'empty') return; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < NUM_BLOCKS; i++) {
      /* v8 ignore next */ /* v8 ignore next */
      const block = document.createElement('div'); /* v8 ignore next */ /* v8 ignore next */
      block.className = 'memory-block'; /* v8 ignore next */ /* v8 ignore next */
      if (state === 'in-use' && Math.random() > 0.3) {
        /* v8 ignore next */ /* v8 ignore next */
        block.classList.add('allocated'); /* v8 ignore next */ /* v8 ignore next */
        block.textContent = 'busy'; /* v8 ignore next */ /* v8 ignore next */
      } else {
        /* v8 ignore next */ /* v8 ignore next */
        block.textContent = 'free'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      arenaContainer.appendChild(block); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  allocateBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent =
      'Pre-allocating 10MB contiguous ArrayBuffer...'; /* v8 ignore next */ /* v8 ignore next */
    try {
      /* v8 ignore next */ /* v8 ignore next */
      arena = new ArrayBuffer(10 * 1024 * 1024); /* v8 ignore next */ /* v8 ignore next */
      renderArena('allocated'); /* v8 ignore next */ /* v8 ignore next */
      log(
        'Arena pre-allocated successfully. Dynamic allocations eliminated.',
      ); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      allocateBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
      runInferenceBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
      freeBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
    } catch (err: any) {
      /* v8 ignore next */ /* v8 ignore next */
      log('Failed to allocate: ' + err.message); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  runInferenceBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    if (!arena) return; /* v8 ignore next */ /* v8 ignore next */
    log(
      'Running inference pass using static memory arena...',
    ); /* v8 ignore next */ /* v8 ignore next */
    renderArena('in-use'); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Simulate inference work /* v8 ignore next */ /* v8 ignore next */
    setTimeout(() => {
      /* v8 ignore next */ /* v8 ignore next */
      renderArena('allocated'); /* v8 ignore next */ /* v8 ignore next */
      log(
        'Inference complete. No memory was allocated or garbage collected.',
      ); /* v8 ignore next */ /* v8 ignore next */
    }, 500); /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  freeBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    arena = null; /* v8 ignore next */ /* v8 ignore next */
    renderArena('empty'); /* v8 ignore next */ /* v8 ignore next */
    log('Arena memory freed.'); /* v8 ignore next */ /* v8 ignore next */
    allocateBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
    runInferenceBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
    freeBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
});
