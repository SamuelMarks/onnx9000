import { describe, it, expect, vi } from 'vitest';
import { initCudaDemo } from './main.js';

describe('demo-cuda', () => {
  it('should initialize and handle click', () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initCudaDemo();
    document.getElementById('run-btn')?.click();
    expect(document.getElementById('output')?.innerText).toBe('Initializing CUDA...');
    vi.runAllTimers();
    expect(document.getElementById('output')?.innerText).toBe(
      'CUDA engine loaded.\nExecution complete: SUCCESS',
    );
    vi.useRealTimers();
  });
});
