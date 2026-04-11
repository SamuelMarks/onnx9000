/* eslint-disable */
import { load } from '@onnx9000/core';
import { InferenceSession } from '@onnx9000/backend-web';

const form = document.getElementById('chat-form') as HTMLFormElement;
const input = document.getElementById('prompt-input') as HTMLInputElement;
const sendBtn = document.getElementById('send-btn') as HTMLButtonElement;
const messagesDiv = document.getElementById('messages') as HTMLElement;

let isGenerating = false;

function addMessage(text: string, sender: 'user' | 'bot') {
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('message', sender);
  msgDiv.textContent = text;
  messagesDiv.appendChild(msgDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  return msgDiv;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (isGenerating || !input.value.trim()) return;

  const prompt = input.value.trim();
  input.value = '';
  addMessage(prompt, 'user');

  isGenerating = true;
  input.disabled = true;
  sendBtn.disabled = true;

  const botMsgDiv = addMessage('...', 'bot');

  try {
    await runLlamaModel(prompt, botMsgDiv);
  } catch (_err) {
    const err = _err instanceof Error ? _err : new Error(String(_err));
    botMsgDiv.textContent = `[Error] ${err.message}`;
  } finally {
    isGenerating = false;
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
});

async function runLlamaModel(prompt: string, element: HTMLElement) {
  // Mock LLM token streaming output
  element.textContent = '';

  const responses = [
    'I am an AI assistant running locally via ONNX9000.',
    ' The underlying engine uses WebGPU for high-throughput matrix multiplication.',
    ' Because I run in your browser, no data is sent to a server.',
    ' How else can I help you today?',
  ];

  for (let i = 0; i < responses.length; i++) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    element.textContent += responses[i];
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }
}
