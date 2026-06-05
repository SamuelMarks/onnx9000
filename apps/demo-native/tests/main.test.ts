import { describe, it, expect, vi } from 'vitest';
import { initNativeCpuDemo } from '../src/main.js';

describe('demo-native', () => {
  it('should run native cpu', () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initNativeCpuDemo();
    document.getElementById('run-btn')?.click();
    expect(document.getElementById('output')?.innerText).toContain('Initializing');
    vi.runAllTimers();
    expect(document.getElementById('output')?.innerText).toContain('Execution complete: SUCCESS');
    vi.useRealTimers();
  });
});
