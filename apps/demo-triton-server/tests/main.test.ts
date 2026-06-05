import { describe, it, expect, vi } from 'vitest';
import { initTritonServerDemo } from '../src/main.js';

describe('demo-triton-server', () => {
  it('should run execution', () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
    initTritonServerDemo();
    document.getElementById('btn-run')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Running');
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain('execution complete');
    vi.useRealTimers();
  });
});
