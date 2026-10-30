// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initModifierDemo } from '../src/main.js';

describe('demo-modifier', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<input id="oldInput" value="input_0"/><input id="newInput" value="x"/><input id="batchSize" value="8"/><button id="btnInit"></button><button id="btnRename"></button><button id="btnBatch"></button><div id="output"></div>';
  });

  it('should initialize, rename inputs, and override batch sizes', () => {
    document.body.innerHTML = '';
    initModifierDemo();

    document.body.innerHTML =
      '<input id="oldInput" value="input_0"/><input id="newInput" value="x"/><input id="batchSize" value="8"/><button id="btnInit"></button><button id="btnRename"></button><button id="btnBatch"></button><div id="output"></div>';
    initModifierDemo();

    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => undefined);
    const btnInit = document.getElementById('btnInit') as HTMLButtonElement;
    const btnRename = document.getElementById('btnRename') as HTMLButtonElement;
    const btnBatch = document.getElementById('btnBatch') as HTMLButtonElement;
    const batchInput = document.getElementById('batchSize') as HTMLInputElement;
    const out = document.getElementById('output') as HTMLDivElement;

    btnRename.click();
    btnBatch.click();
    expect(alertSpy).toHaveBeenCalledTimes(2);

    btnInit.click();
    expect(out.textContent).toContain('input_0 [1,3,224,224]');

    btnRename.click();
    expect(out.textContent).toContain('x [1,3,224,224]');

    btnBatch.click();
    expect(out.textContent).toContain('x [8,3,224,224]');

    batchInput.value = 'N';
    btnBatch.click();
    expect(out.textContent).toContain('x [N,3,224,224]');

    alertSpy.mockRestore();
  });
});
