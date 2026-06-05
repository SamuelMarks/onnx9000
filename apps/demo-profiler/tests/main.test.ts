import { describe, it, expect, vi } from 'vitest';
import { initProfilerDemo } from '../src/main.js';

describe('demo-profiler', () => {
  it('should run profiler', () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
    initProfilerDemo();
    document.getElementById('btn-run')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Initializing profiler');
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain('[OK] execution complete');
    vi.useRealTimers();
  });
});
