/* eslint-disable */
import { load } from '@onnx9000/core';
import { InferenceSession } from '@onnx9000/backend-web';

const logEl = document.getElementById('log') as HTMLElement;
const recordBtn = document.getElementById('record-btn') as HTMLButtonElement;
const clearBtn = document.getElementById('clear-btn') as HTMLButtonElement;

let isRecording = false;
let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];

// Stubs for models
let whisperSession: InferenceSession | null = null;
let llmSession: InferenceSession | null = null;

function appendLog(msg: string) {
  logEl.textContent += '\n' + msg;
  logEl.scrollTop = logEl.scrollHeight;
}

clearBtn.addEventListener('click', () => {
  logEl.textContent = 'Log cleared.';
});

async function initModels() {
  appendLog('[System] Initializing WebGPU backend...');
  try {
    // In a real app we would load .onnx files here
    // e.g. whisperSession = await InferenceSession.create('whisper-tiny.onnx', { executionProviders: ['webgpu'] });
    appendLog('[System] WebGPU backend ready (Mocked). Models loaded.');
    recordBtn.disabled = false;
  } catch (_err) {
    const err = _err instanceof Error ? _err : new Error(String(_err));
    appendLog(`[Error] Failed to initialize: ${err.message}`);
  }
}

recordBtn.addEventListener('click', async () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      audioChunks.push(e.data);
    };

    mediaRecorder.onstop = processAudio;

    mediaRecorder.start();
    isRecording = true;
    recordBtn.textContent = 'Stop Recording';
    recordBtn.classList.add('recording');
    appendLog('[Mic] Recording started...');
  } catch (_err) {
    const err = _err instanceof Error ? _err : new Error(String(_err));
    appendLog(`[Mic Error] Could not access microphone: ${err.message}`);
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  const stream = mediaRecorder?.stream;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  isRecording = false;
  recordBtn.textContent = 'Start Recording';
  recordBtn.classList.remove('recording');
  appendLog('[Mic] Recording stopped. Processing...');
}

async function processAudio() {
  appendLog('[Whisper] Transcribing audio buffer via WebGPU...');
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  const arrayBuffer = await blob.arrayBuffer();

  // Mock transcription delay
  await new Promise((resolve) => setTimeout(resolve, 1500));

  // Dummy text
  const text = 'Hello, can you explain what WebGPU is?';
  appendLog(`[User] "${text}"`);

  await runLLM(text);
}

async function runLLM(prompt: string) {
  appendLog(`[LLM] Generating response for: "${prompt}"...`);

  // Mock LLM generation
  const responseTokens = [
    'WebGPU',
    'is',
    'a',
    'modern',
    'graphics',
    'API',
    'that',
    'brings',
    'low-level',
    'access',
    'to',
    'GPU',
    'hardware',
    'directly',
    'in',
    'the',
    'browser,',
    'enabling',
    'high-performance',
    'machine',
    'learning',
    'inference.',
  ];

  appendLog('[Assistant] ');

  for (const token of responseTokens) {
    await new Promise((resolve) => setTimeout(resolve, 100)); // streaming effect
    logEl.textContent += token + ' ';
    logEl.scrollTop = logEl.scrollHeight;
  }
  appendLog('\n[System] Generation complete.');
}

// Start init process
initModels();
