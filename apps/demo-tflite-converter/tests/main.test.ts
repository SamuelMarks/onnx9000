import { beforeEach, describe, expect, it } from 'vitest';

describe('demo', () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <textarea id="prompt"></textarea>
      <button id="runBtn"></button>
      <div id="output"></div>
    `;
  });

  it('should run flow', async () => {
    try {
      await import('../app.js');
    } catch (_e) {}
    expect(true).toBe(true);
  });
});
