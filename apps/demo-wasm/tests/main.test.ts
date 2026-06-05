import { describe, it, expect, vi } from 'vitest';
import { initWasmDemo } from '../src/main.js';

describe('demo-wasm', () => {
  it('should run wasm demo', () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initWasmDemo();
    document.getElementById('run-btn')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Initializing');
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain('SUCCESS');
    vi.useRealTimers();
  });
});
