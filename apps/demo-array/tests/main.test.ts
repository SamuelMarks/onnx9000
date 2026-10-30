// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initArrayDemo } from '../src/main.js';

describe('demo-array', () => {
  beforeEach(() => {
    document.body.innerHTML = '<button id="run-btn"></button><div id="array-output"></div>';
  });

  it('should execute eager and lazy array operations', () => {
    document.body.innerHTML = '';
    initArrayDemo();

    document.body.innerHTML = '<button id="run-btn"></button><div id="array-output"></div>';
    initArrayDemo();

    const btn = document.getElementById('run-btn') as HTMLButtonElement;
    const out = document.getElementById('array-output') as HTMLElement;

    btn.click();
    expect(out.innerText).toContain('Success! The Array API is fully functional.');
    expect(btn.disabled).toBe(false);
  });
});
