// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initProfilerDemo } from '../src/main.js';

describe('demo-profiler', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
  });

  it('should initialize and run profiler demo', async () => {
    document.body.innerHTML = '';
    initProfilerDemo();

    document.body.innerHTML = '<button id="btn-run"></button>';
    initProfilerDemo();
    document.getElementById('btn-run')?.click();

    document.body.innerHTML = '<button id="btn-run"></button><div id="output"></div>';
    initProfilerDemo();

    const btn = document.getElementById('btn-run') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.textContent).toContain('Initializing profiler...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.textContent).toContain('[OK] execution complete');
  });
});
