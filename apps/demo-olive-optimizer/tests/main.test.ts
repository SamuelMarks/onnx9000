import { describe, it, expect, vi } from 'vitest';
import { initOliveOptimizerDemo } from '../src/main.js';

describe('demo-olive-optimizer', () => {
  it('should run optimizer', () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
    initOliveOptimizerDemo();
    document.getElementById('btn-run')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Running...');
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain(
      '[OK] Olive Optimizer execution complete.',
    );
    vi.useRealTimers();
  });
});
