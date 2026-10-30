// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initCompileDemo } from '../src/main.js';

describe('demo-compile', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="btn-compile"></button><div id="output"></div>';
  });

  it('should initialize and run compile demo', async () => {
    document.body.innerHTML = '';
    initCompileDemo();

    document.body.innerHTML = '<button id="btn-compile"></button>';
    initCompileDemo();
    document.getElementById('btn-compile')?.click();

    document.body.innerHTML = '<button id="btn-compile"></button><div id="output"></div>';
    initCompileDemo();

    const btn = document.getElementById('btn-compile') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLElement;

    btn.click();
    expect(out.textContent).toContain('Compiling...');

    await new Promise((r) => setTimeout(r, 100));
    expect(out.textContent).toContain('[OK] AOT Compilation finished: model.bin');
  });
});
