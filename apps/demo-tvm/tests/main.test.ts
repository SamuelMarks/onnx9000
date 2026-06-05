import { describe, it, expect, vi } from 'vitest';
import { initTvmDemo } from '../src/main.js';

describe('demo-tvm', () => {
  it('should run tvm conversion', () => {
    document.body.innerHTML = '<button id="convert-btn"></button><div id="output"></div>';
    initTvmDemo();
    document.getElementById('convert-btn')?.click();
    expect(document.getElementById('output')?.textContent).toContain('def @main');
  });
});
