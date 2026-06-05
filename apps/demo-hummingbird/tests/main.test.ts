import { describe, it, expect, vi } from 'vitest';
import { initHummingbirdDemo } from '../src/main.js';

describe('demo-hummingbird', () => {
  it('should run hummingbird transpiler', async () => {
    vi.useFakeTimers();
    document.body.innerHTML =
      '<button id="transpile-btn"></button><div id="transpiler-output"></div>';
    initHummingbirdDemo();
    document.getElementById('transpile-btn')?.click();

    for (let i = 0; i < 5; i++) {
      vi.runAllTimers();
      await new Promise((r) => process.nextTick(r));
    }

    expect(document.getElementById('transpiler-output')?.innerText).toContain(
      'Transpilation successful',
    );
    vi.useRealTimers();
  });
});
