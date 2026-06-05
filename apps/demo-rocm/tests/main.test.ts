import { describe, it, expect, vi } from 'vitest';
import { initRocmDemo } from '../src/main.js';

describe('demo-rocm', () => {
  it('should run rocm', () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initRocmDemo();
    document.getElementById('run-btn')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Initializing ROCm');
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain('Execution complete: SUCCESS');
    vi.useRealTimers();
  });
});
