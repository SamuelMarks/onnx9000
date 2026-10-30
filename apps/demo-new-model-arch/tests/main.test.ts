// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initNewModelArchDemo } from '../src/main.js';

describe('demo-new-model-arch', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<select id="archSelect"><option value="mamba">mamba</option><option value="rwkv">rwkv</option><option value="mixtral">mixtral</option></select><button id="parseBtn"></button><button id="resetBtn"></button><div id="output"></div>';
  });

  it('should generate IR for mamba, rwkv, and mixtral architectures', async () => {
    document.body.innerHTML = '';
    initNewModelArchDemo();

    document.body.innerHTML =
      '<select id="archSelect"><option value="mamba">mamba</option><option value="rwkv">rwkv</option><option value="mixtral">mixtral</option></select><button id="parseBtn"></button><button id="resetBtn"></button><div id="output"></div>';
    initNewModelArchDemo();

    const select = document.getElementById('archSelect') as HTMLSelectElement;
    const parseBtn = document.getElementById('parseBtn') as HTMLButtonElement;
    const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
    const out = document.getElementById('output') as HTMLDivElement;

    parseBtn.click();
    await new Promise((r) => setTimeout(r, 180));
    expect(out.textContent).toContain('SelectiveScan');

    resetBtn.click();
    expect(out.textContent).toContain('Select an architecture');

    select.value = 'rwkv';
    parseBtn.click();
    await new Promise((r) => setTimeout(r, 180));
    expect(out.textContent).toContain('TimeMix');

    resetBtn.click();
    select.value = 'mixtral';
    parseBtn.click();
    await new Promise((r) => setTimeout(r, 180));
    expect(out.textContent).toContain('Router Gate');

    resetBtn.click();
    select.innerHTML = '<option value=""></option>';
    select.value = '';
    parseBtn.click();
    await new Promise((r) => setTimeout(r, 180));
    expect(out.textContent).toContain('SelectiveScan');
  });
});
