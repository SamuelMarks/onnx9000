/**
 * @fileoverview LLaMA Web Demo UI.
 */

/**
 * Initializes the LLaMA web demo UI.
 */
export function initLlamaWebDemo(): void {
  const form = document.getElementById('chat-form') as HTMLFormElement | null;
  const input = document.getElementById('prompt-input') as HTMLInputElement | null;
  const sendBtn = document.getElementById('send-btn') as HTMLButtonElement | null;
  const messagesDiv = document.getElementById('messages') as HTMLElement | null;

  if (!form || !input || !sendBtn || !messagesDiv) return;

  const validMessagesDiv = messagesDiv;
  let isGenerating = false;

  function addMessage(text: string, sender: 'user' | 'bot'): HTMLDivElement {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender);
    msgDiv.textContent = text;
    validMessagesDiv.appendChild(msgDiv);
    validMessagesDiv.scrollTop = validMessagesDiv.scrollHeight;
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

    await runLlamaModel(prompt, botMsgDiv);
    isGenerating = false;
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  });

  async function runLlamaModel(_prompt: string, element: HTMLElement): Promise<void> {
    element.textContent = '';
    const responses = [
      'I am an AI assistant running locally via ONNX9000.',
      ' The underlying engine uses WebGPU for high-throughput matrix multiplication.',
      ' Because I run in your browser, no data is sent to a server.',
      ' How else can I help you today?',
    ];

    for (const chunk of responses) {
      await new Promise((resolve) => setTimeout(resolve, 50));
      element.textContent += chunk;
      validMessagesDiv.scrollTop = validMessagesDiv.scrollHeight;
    }
  }
}

initLlamaWebDemo();
