// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initServeDemo } from '../src/main.js';

describe('demo-serve', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<button id="start-btn"></button><button id="req-btn"></button><div id="server-output"></div>';
  });

  it('should start edge server and handle mock KServe inference request', async () => {
    document.body.innerHTML = '';
    initServeDemo();

    document.body.innerHTML =
      '<button id="start-btn"></button><button id="req-btn"></button><div id="server-output"></div>';
    initServeDemo();

    const startBtn = document.getElementById('start-btn') as HTMLButtonElement;
    const reqBtn = document.getElementById('req-btn') as HTMLButtonElement;
    const out = document.getElementById('server-output') as HTMLElement;

    reqBtn.click();

    startBtn.click();
    expect(out.innerText).toContain('Server initialized with KServe');

    reqBtn.click();
    await new Promise((r) => setTimeout(r, 50));
    expect(out.innerText).toContain('Success! Edge routing is fully functional in-browser.');
  });
});
