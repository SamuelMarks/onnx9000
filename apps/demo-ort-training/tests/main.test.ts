import { describe, it, expect, vi } from 'vitest';
import { initOrtTrainingDemo } from '../src/main.js';

describe('demo-ort-training', () => {
  it('should run ort training', () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
    initOrtTrainingDemo();
    document.getElementById('btn-run')?.click();
    expect(document.getElementById('output')?.textContent).toContain('Running');
    vi.runAllTimers();
    expect(document.getElementById('output')?.textContent).toContain(
      '[OK] ORT Training execution complete.',
    );
    vi.useRealTimers();
  });
});
