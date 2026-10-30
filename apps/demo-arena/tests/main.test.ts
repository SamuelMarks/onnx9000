// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initArenaDemo } from '../src/main.js';

describe('demo-arena', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<button id="run-profiler"></button><div id="peak-mem"></div><button id="refresh-arena"></button><div id="blocks"></div>';
  });

  it('should update peak memory and render memory blocks', () => {
    document.body.innerHTML = '';
    initArenaDemo();

    document.body.innerHTML =
      '<button id="run-profiler"></button><button id="refresh-arena"></button>';
    initArenaDemo();
    document.getElementById('run-profiler')?.click();
    document.getElementById('refresh-arena')?.click();

    document.body.innerHTML =
      '<button id="run-profiler"></button><div id="peak-mem"></div><button id="refresh-arena"></button><div id="blocks"></div>';
    initArenaDemo();

    document.getElementById('run-profiler')?.click();
    expect(Number(document.getElementById('peak-mem')?.textContent)).toBeGreaterThanOrEqual(50);

    document.getElementById('refresh-arena')?.click();
    expect(document.getElementById('blocks')?.children.length).toBeGreaterThanOrEqual(5);
  });
});
