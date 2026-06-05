import { describe, it, expect, vi } from 'vitest';
import { initLlamaWebDemo } from '../src/main.js';

describe('demo-llama-web', () => {
  it('should run chat', async () => {
    vi.useFakeTimers();
    document.body.innerHTML = `
      <form id="chat-form">
        <input id="prompt-input" value="hello" />
        <button id="send-btn"></button>
      </form>
      <div id="messages"></div>
    `;
    initLlamaWebDemo();

    document.getElementById('chat-form')?.dispatchEvent(new Event('submit', { cancelable: true }));

    for (let i = 0; i < 10; i++) {
      vi.runAllTimers();
      await new Promise((r) => process.nextTick(r));
    }

    expect(document.getElementById('messages')?.textContent).toContain(
      'AI assistant running locally',
    );
    vi.useRealTimers();
  });
});
