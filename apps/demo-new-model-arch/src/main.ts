/**
 * @fileoverview New Model Architecture Builder Demo UI.
 */
import { Graph } from '@onnx9000/core';

/**
 * Initializes the new model architecture demo UI.
 */
export function initNewModelArchDemo(): void {
  const parseBtn = document.getElementById('parseBtn') as HTMLButtonElement | null;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement | null;
  const outputDiv = document.getElementById('output') as HTMLDivElement | null;
  const archSelect = document.getElementById('archSelect') as HTMLSelectElement | null;

  if (!parseBtn || !resetBtn || !outputDiv) return;

  const log = (msg: string) => {
    outputDiv.textContent += `${msg}
`;
  };

  parseBtn.addEventListener('click', () => {
    parseBtn.disabled = true;
    outputDiv.textContent = '';

    const arch = archSelect?.value || 'mamba';
    log(`Initializing architecture builder for: ${arch.toUpperCase()}...`);

    setTimeout(() => {
      const g = new Graph(`${arch}_graph`);
      log('Instantiating core blocks...');

      setTimeout(() => {
        if (arch === 'mamba') {
          log(' - Added SelectiveScan (SSM) nodes');
          log(' - Added 1D Depthwise Convolution');
        } else if (arch === 'rwkv') {
          log(' - Added TimeMix and ChannelMix blocks');
          log(' - Configured WKV linear attention');
        } else {
          log(' - Added Router Gate (Top-2) and 8 Expert MLPs');
        }

        setTimeout(() => {
          log(`
Generated IR JSON:`);
          log(JSON.stringify(g, null, 2));
          resetBtn.disabled = false;
        }, 40);
      }, 40);
    }, 40);
  });

  resetBtn.addEventListener('click', () => {
    outputDiv.textContent = `Select an architecture and click "Generate Graph IR".
`;
    parseBtn.disabled = false;
    resetBtn.disabled = true;
  });
}

initNewModelArchDemo();
