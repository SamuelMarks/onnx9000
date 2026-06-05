import { describe, it, expect, vi } from 'vitest';
import { initModifierDemo } from '../src/main.js';

// mock alert
global.alert = vi.fn();

describe('demo-modifier', () => {
  it('should initialize and modify', () => {
    document.body.innerHTML = `
      <button id="btnInit"></button>
      <button id="btnRename"></button>
      <button id="btnBatch"></button>
      <div id="output"></div>
      <input id="oldInput" value="input_0" />
      <input id="newInput" value="input_1" />
      <input id="batchSize" value="4" />
    `;
    initModifierDemo();

    document.getElementById('btnInit')?.click();
    expect(document.getElementById('output')?.textContent).toContain('input_0');

    document.getElementById('btnRename')?.click();
    expect(document.getElementById('output')?.textContent).toContain('input_1');

    document.getElementById('btnBatch')?.click();
    expect(document.getElementById('output')?.textContent).toContain('4,3,224,224');
  });
});
