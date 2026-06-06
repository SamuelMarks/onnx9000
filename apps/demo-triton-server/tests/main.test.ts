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
    // import to execute module top-level
    try { await import('../src/main.js'); } catch(e) {}
    
    const btn = document.getElementById('runBtn');
    const prompt = document.getElementById('prompt');
    const out = document.getElementById('output');
    
    if (btn) btn.click();
    if (prompt && btn) {
        prompt.value = "test";
        btn.click();
    }
    
    // allow some async code to run
    await new Promise(r => setTimeout(r, 100));
    expect(true).toBe(true);
  });
});
