// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { initLlamaWebDemo } from '../src/main.js';

describe('demo-llama-web', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<form id="chat-form"><input id="prompt-input"/><button id="send-btn"></button><div id="messages"></div></form>';
  });

  it('should run flow and stream response messages', async () => {
    document.body.innerHTML = '';
    initLlamaWebDemo();

    document.body.innerHTML =
      '<form id="chat-form"><input id="prompt-input"/><button id="send-btn"></button><div id="messages"></div></form>';
    initLlamaWebDemo();

    const form = document.getElementById('chat-form') as HTMLFormElement;
    const input = document.getElementById('prompt-input') as HTMLInputElement;
    const messages = document.getElementById('messages') as HTMLElement;

    form.dispatchEvent(new Event('submit', { cancelable: true }));

    input.value = 'hello';
    form.dispatchEvent(new Event('submit', { cancelable: true }));

    input.value = 'ignored';
    form.dispatchEvent(new Event('submit', { cancelable: true }));

    await new Promise((r) => setTimeout(r, 300));
    expect(messages.textContent).toContain('How else can I help you today?');
  });
});
