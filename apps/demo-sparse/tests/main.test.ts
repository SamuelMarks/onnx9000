// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initSparseDemo } from '../src/main.js';

describe('demo-sparse', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="prune-btn"></button><div id="sparse-output"></div>';
  });

  it('should execute sparse pruning flow and update output', async () => {
    document.body.innerHTML = '';
    initSparseDemo();

    document.body.innerHTML = '<button id="prune-btn"></button><div id="sparse-output"></div>';
    initSparseDemo();

    const btn = document.getElementById('prune-btn') as HTMLButtonElement;
    const out = document.getElementById('sparse-output') as HTMLElement;

    btn.click();
    await new Promise((r) => setTimeout(r, 150));

    expect(out.innerText).toContain('Sparsification successful!');
    expect(btn.disabled).toBe(true);
  });
});
