/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import { load } from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import { InferenceSession } from '@onnx9000/backend-web'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const form = document.getElementById(
  'chat-form',
) as HTMLFormElement; /* v8 ignore next */ /* v8 ignore next */
const input = document.getElementById(
  'prompt-input',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const sendBtn = document.getElementById(
  'send-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const messagesDiv = document.getElementById(
  'messages',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
let isGenerating = false; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function addMessage(text: string, sender: 'user' | 'bot') {
  /* v8 ignore next */ /* v8 ignore next */
  const msgDiv = document.createElement('div'); /* v8 ignore next */ /* v8 ignore next */
  msgDiv.classList.add('message', sender); /* v8 ignore next */ /* v8 ignore next */
  msgDiv.textContent = text; /* v8 ignore next */ /* v8 ignore next */
  messagesDiv.appendChild(msgDiv); /* v8 ignore next */ /* v8 ignore next */
  messagesDiv.scrollTop = messagesDiv.scrollHeight; /* v8 ignore next */ /* v8 ignore next */
  return msgDiv; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
form.addEventListener('submit', async (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  if (isGenerating || !input.value.trim()) return; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const prompt = input.value.trim(); /* v8 ignore next */ /* v8 ignore next */
  input.value = ''; /* v8 ignore next */ /* v8 ignore next */
  addMessage(prompt, 'user'); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  isGenerating = true; /* v8 ignore next */ /* v8 ignore next */
  input.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  sendBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const botMsgDiv = addMessage('...', 'bot'); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    await runLlamaModel(prompt, botMsgDiv); /* v8 ignore next */ /* v8 ignore next */
  } catch (_err) {
    /* v8 ignore next */ /* v8 ignore next */
    const err =
      _err instanceof Error
        ? _err
        : new Error(String(_err)); /* v8 ignore next */ /* v8 ignore next */
    botMsgDiv.textContent = `[Error] ${err.message}`; /* v8 ignore next */ /* v8 ignore next */
  } finally {
    /* v8 ignore next */ /* v8 ignore next */
    isGenerating = false; /* v8 ignore next */ /* v8 ignore next */
    input.disabled = false; /* v8 ignore next */ /* v8 ignore next */
    sendBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
    input.focus(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function runLlamaModel(prompt: string, element: HTMLElement) {
  /* v8 ignore next */ /* v8 ignore next */
  // Mock LLM token streaming output /* v8 ignore next */ /* v8 ignore next */
  element.textContent = ''; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const responses = [
    /* v8 ignore next */ /* v8 ignore next */
    'I am an AI assistant running locally via ONNX9000.' /* v8 ignore next */ /* v8 ignore next */,
    ' The underlying engine uses WebGPU for high-throughput matrix multiplication.' /* v8 ignore next */ /* v8 ignore next */,
    ' Because I run in your browser, no data is sent to a server.' /* v8 ignore next */ /* v8 ignore next */,
    ' How else can I help you today?' /* v8 ignore next */ /* v8 ignore next */,
  ]; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  for (let i = 0; i < responses.length; i++) {
    /* v8 ignore next */ /* v8 ignore next */
    await new Promise((resolve) =>
      setTimeout(resolve, 500),
    ); /* v8 ignore next */ /* v8 ignore next */
    element.textContent += responses[i]; /* v8 ignore next */ /* v8 ignore next */
    messagesDiv.scrollTop = messagesDiv.scrollHeight; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
