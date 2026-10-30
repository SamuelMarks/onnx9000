// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initWebnnDemo } from '../src/main.js';

describe('demo-webnn-polyfill', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="webnn-output"></div>';
  });

  it('should execute WebNN graph and handle missing navigator.ml or errors', async () => {
    document.body.innerHTML = '';
    initWebnnDemo();

    document.body.innerHTML = '<button id="run-btn"></button><div id="webnn-output"></div>';
    initWebnnDemo();

    const runBtn = document.getElementById('run-btn') as HTMLButtonElement;
    const out = document.getElementById('webnn-output') as HTMLElement;

    runBtn.click();
    await new Promise((r) => setTimeout(r, 50));
    expect(out.innerText).toContain('Success! WebNN API execution complete.');

    const winRecord = window as never as Record<string, unknown>;
    const savedBuilder = winRecord.MLGraphBuilder;
    delete winRecord.MLGraphBuilder;
    runBtn.click();
    await new Promise((r) => setTimeout(r, 50));
    expect(out.innerText).toContain('Success! WebNN API execution complete.');
    winRecord.MLGraphBuilder = savedBuilder;

    const navRecord = navigator as never as Record<
      string,
      { createContext: () => Promise<unknown> } | undefined
    >;
    const savedMl = navRecord.ml;
    navRecord.ml = {
      createContext: vi.fn().mockRejectedValueOnce('Non-error rejection'),
    };
    runBtn.click();
    await new Promise((r) => setTimeout(r, 50));
    expect(out.innerText).toContain('Error: Non-error rejection');

    delete navRecord.ml;
    runBtn.click();
    await new Promise((r) => setTimeout(r, 50));
    expect(out.innerText).toContain('Error: WebNN polyfill failed to inject navigator.ml');

    navRecord.ml = savedMl;
  });
});
