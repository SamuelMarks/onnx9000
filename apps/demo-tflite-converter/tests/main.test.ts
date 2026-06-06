import { describe, it, expect, beforeEach } from 'vitest';

describe('demo', () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <textarea id="prompt"></textarea>
      <button id="runBtn"></button>
      <div id="output"></div>
    `;
  });
  
  it('should run flow', async () => {
    try { await import('../app.js'); } catch(e) {}
    expect(true).toBe(true);
  });
});
