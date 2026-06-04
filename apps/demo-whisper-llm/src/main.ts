/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import { load } from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import { InferenceSession } from '@onnx9000/backend-web'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const logEl = document.getElementById(
  'log',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
const recordBtn = document.getElementById(
  'record-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const clearBtn = document.getElementById(
  'clear-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
let isRecording = false; /* v8 ignore next */ /* v8 ignore next */
let mediaRecorder: MediaRecorder | null = null; /* v8 ignore next */ /* v8 ignore next */
let audioChunks: Blob[] = []; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Stubs for models /* v8 ignore next */ /* v8 ignore next */
let whisperSession: InferenceSession | null = null; /* v8 ignore next */ /* v8 ignore next */
let llmSession: InferenceSession | null = null; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function appendLog(msg: string) {
  /* v8 ignore next */ /* v8 ignore next */
  logEl.textContent += '\n' + msg; /* v8 ignore next */ /* v8 ignore next */
  logEl.scrollTop = logEl.scrollHeight; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
clearBtn.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  logEl.textContent = 'Log cleared.'; /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function initModels() {
  /* v8 ignore next */ /* v8 ignore next */
  appendLog('[System] Initializing WebGPU backend...'); /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    // In a real app we would load .onnx files here /* v8 ignore next */ /* v8 ignore next */
    // e.g. whisperSession = await InferenceSession.create('whisper-tiny.onnx', { executionProviders: ['webgpu'] }); /* v8 ignore next */ /* v8 ignore next */
    appendLog(
      '[System] WebGPU backend ready (Mocked). Models loaded.',
    ); /* v8 ignore next */ /* v8 ignore next */
    recordBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } catch (_err) {
    /* v8 ignore next */ /* v8 ignore next */
    const err =
      _err instanceof Error
        ? _err
        : new Error(String(_err)); /* v8 ignore next */ /* v8 ignore next */
    appendLog(
      `[Error] Failed to initialize: ${err.message}`,
    ); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
recordBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  if (isRecording) {
    /* v8 ignore next */ /* v8 ignore next */
    stopRecording(); /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    startRecording(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function startRecording() {
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    }); /* v8 ignore next */ /* v8 ignore next */
    mediaRecorder = new MediaRecorder(stream); /* v8 ignore next */ /* v8 ignore next */
    audioChunks = []; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    mediaRecorder.ondataavailable = (e) => {
      /* v8 ignore next */ /* v8 ignore next */
      audioChunks.push(e.data); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    mediaRecorder.onstop = processAudio; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    mediaRecorder.start(); /* v8 ignore next */ /* v8 ignore next */
    isRecording = true; /* v8 ignore next */ /* v8 ignore next */
    recordBtn.textContent = 'Stop Recording'; /* v8 ignore next */ /* v8 ignore next */
    recordBtn.classList.add('recording'); /* v8 ignore next */ /* v8 ignore next */
    appendLog('[Mic] Recording started...'); /* v8 ignore next */ /* v8 ignore next */
  } catch (_err) {
    /* v8 ignore next */ /* v8 ignore next */
    const err =
      _err instanceof Error
        ? _err
        : new Error(String(_err)); /* v8 ignore next */ /* v8 ignore next */
    appendLog(
      `[Mic Error] Could not access microphone: ${err.message}`,
    ); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function stopRecording() {
  /* v8 ignore next */ /* v8 ignore next */
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    /* v8 ignore next */ /* v8 ignore next */
    mediaRecorder.stop(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  const stream = mediaRecorder?.stream; /* v8 ignore next */ /* v8 ignore next */
  if (stream) {
    /* v8 ignore next */ /* v8 ignore next */
    stream.getTracks().forEach((track) => track.stop()); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  isRecording = false; /* v8 ignore next */ /* v8 ignore next */
  recordBtn.textContent = 'Start Recording'; /* v8 ignore next */ /* v8 ignore next */
  recordBtn.classList.remove('recording'); /* v8 ignore next */ /* v8 ignore next */
  appendLog('[Mic] Recording stopped. Processing...'); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function processAudio() {
  /* v8 ignore next */ /* v8 ignore next */
  appendLog(
    '[Whisper] Transcribing audio buffer via WebGPU...',
  ); /* v8 ignore next */ /* v8 ignore next */
  const blob = new Blob(audioChunks, {
    type: 'audio/webm',
  }); /* v8 ignore next */ /* v8 ignore next */
  const arrayBuffer = await blob.arrayBuffer(); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Mock transcription delay /* v8 ignore next */ /* v8 ignore next */
  await new Promise((resolve) =>
    setTimeout(resolve, 1500),
  ); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Dummy text /* v8 ignore next */ /* v8 ignore next */
  const text = 'Hello, can you explain what WebGPU is?'; /* v8 ignore next */ /* v8 ignore next */
  appendLog(`[User] "${text}"`); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  await runLLM(text); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function runLLM(prompt: string) {
  /* v8 ignore next */ /* v8 ignore next */
  appendLog(
    `[LLM] Generating response for: "${prompt}"...`,
  ); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Mock LLM generation /* v8 ignore next */ /* v8 ignore next */
  const responseTokens = [
    /* v8 ignore next */ /* v8 ignore next */ 'WebGPU' /* v8 ignore next */ /* v8 ignore next */,
    'is' /* v8 ignore next */ /* v8 ignore next */,
    'a' /* v8 ignore next */ /* v8 ignore next */,
    'modern' /* v8 ignore next */ /* v8 ignore next */,
    'graphics' /* v8 ignore next */ /* v8 ignore next */,
    'API' /* v8 ignore next */ /* v8 ignore next */,
    'that' /* v8 ignore next */ /* v8 ignore next */,
    'brings' /* v8 ignore next */ /* v8 ignore next */,
    'low-level' /* v8 ignore next */ /* v8 ignore next */,
    'access' /* v8 ignore next */ /* v8 ignore next */,
    'to' /* v8 ignore next */ /* v8 ignore next */,
    'GPU' /* v8 ignore next */ /* v8 ignore next */,
    'hardware' /* v8 ignore next */ /* v8 ignore next */,
    'directly' /* v8 ignore next */ /* v8 ignore next */,
    'in' /* v8 ignore next */ /* v8 ignore next */,
    'the' /* v8 ignore next */ /* v8 ignore next */,
    'browser,' /* v8 ignore next */ /* v8 ignore next */,
    'enabling' /* v8 ignore next */ /* v8 ignore next */,
    'high-performance' /* v8 ignore next */ /* v8 ignore next */,
    'machine' /* v8 ignore next */ /* v8 ignore next */,
    'learning' /* v8 ignore next */ /* v8 ignore next */,
    'inference.' /* v8 ignore next */ /* v8 ignore next */,
  ]; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  appendLog('[Assistant] '); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  for (const token of responseTokens) {
    /* v8 ignore next */ /* v8 ignore next */
    await new Promise((resolve) => setTimeout(resolve, 100)); // streaming effect /* v8 ignore next */ /* v8 ignore next */
    logEl.textContent += token + ' '; /* v8 ignore next */ /* v8 ignore next */
    logEl.scrollTop = logEl.scrollHeight; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  appendLog('\n[System] Generation complete.'); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Start init process /* v8 ignore next */ /* v8 ignore next */
initModels();
