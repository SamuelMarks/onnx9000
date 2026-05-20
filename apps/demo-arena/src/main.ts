document.getElementById('run-profiler')?.addEventListener('click', () => {
  const peakMem = document.getElementById('peak-mem');
  if (peakMem) peakMem.textContent = (Math.random() * 100 + 50).toFixed(2);
});

document.getElementById('refresh-arena')?.addEventListener('click', () => {
  const blocksContainer = document.getElementById('blocks');
  if (blocksContainer) {
    blocksContainer.innerHTML = '';
    const numBlocks = Math.floor(Math.random() * 10) + 5;
    for (let i = 0; i < numBlocks; i++) {
      const block = document.createElement('div');
      block.className = 'memory-block';
      block.textContent = `${(Math.random() * 10).toFixed(1)} MB`;
      blocksContainer.appendChild(block);
    }
  }
});
