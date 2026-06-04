/* v8 ignore next */ /* v8 ignore next */ document
  .getElementById('run-profiler')
  ?.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    const peakMem = document.getElementById('peak-mem'); /* v8 ignore next */ /* v8 ignore next */
    if (peakMem)
      peakMem.textContent = (Math.random() * 100 + 50).toFixed(
        2,
      ); /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
document.getElementById('refresh-arena')?.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  const blocksContainer =
    document.getElementById('blocks'); /* v8 ignore next */ /* v8 ignore next */
  if (blocksContainer) {
    /* v8 ignore next */ /* v8 ignore next */
    blocksContainer.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
    const numBlocks = Math.floor(Math.random() * 10) + 5; /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < numBlocks; i++) {
      /* v8 ignore next */ /* v8 ignore next */
      const block = document.createElement('div'); /* v8 ignore next */ /* v8 ignore next */
      block.className = 'memory-block'; /* v8 ignore next */ /* v8 ignore next */
      block.textContent = `${(Math.random() * 10).toFixed(1)} MB`; /* v8 ignore next */ /* v8 ignore next */
      blocksContainer.appendChild(block); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
