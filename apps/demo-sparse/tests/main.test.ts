import { describe, it, expect, vi } from 'vitest';
import { initSparseDemo } from '../src/main.js';

describe('demo-sparse', () => {
  it('should run sparsification', async () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="prune-btn"></button><div id="sparse-output"></div>';
    initSparseDemo();
    document.getElementById('prune-btn')?.click();

    for (let i = 0; i < 5; i++) {
      vi.runAllTimers();
      await new Promise((r) => process.nextTick(r));
    }

    expect(document.getElementById('sparse-output')?.textContent).toContain(
      'Sparsification successful!',
    );
    vi.useRealTimers();
  });
});
