/**
 * @fileoverview Whisper -> LLM Speech Pipeline Demo UI.
 */

/**
 * Initializes the Whisper -> LLM demo UI.
 */
export function initWhisperLlmDemo(): void {
  const logEl = document.getElementById('log') as HTMLElement | null;
  const recordBtn = document.getElementById('record-btn') as HTMLButtonElement | null;
  const clearBtn = document.getElementById('clear-btn') as HTMLButtonElement | null;

  if (!logEl || !recordBtn || !clearBtn) return;

  const validLogEl = logEl;
  const validRecordBtn = recordBtn;
  let isRecording = false;
  let mediaRecorder: MediaRecorder | null = null;
  let audioChunks: Blob[] = [];

  function appendLog(msg: string): void {
    validLogEl.textContent += `
${msg}`;
    validLogEl.scrollTop = validLogEl.scrollHeight;
  }

  clearBtn.addEventListener('click', () => {
    validLogEl.textContent = 'Log cleared.';
  });

  function initModels(): void {
    appendLog('[System] Initializing WebGPU backend...');
    appendLog('[System] WebGPU backend ready (Mocked). Models loaded.');
    validRecordBtn.disabled = false;
  }

  validRecordBtn.addEventListener('click', () => {
    if (isRecording) {
      stopRecording();
    } else {
      void startRecording();
    }
  });

  async function startRecording(): Promise<void> {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => {
        audioChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        void processAudio();
      };

      mediaRecorder.start();
      isRecording = true;
      validRecordBtn.textContent = 'Stop Recording';
      validRecordBtn.classList.add('recording');
      appendLog('[Mic] Recording started...');
    } catch (_err) {
      const err = _err instanceof Error ? _err : new Error(String(_err));
      appendLog(`[Mic Error] Could not access microphone: ${err.message}`);
    }
  }

  function stopRecording(): void {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
    const stream = mediaRecorder?.stream;
    if (stream) {
      stream.getTracks().forEach((track) => {
        track.stop();
      });
    }
    isRecording = false;
    validRecordBtn.textContent = 'Start Recording';
    validRecordBtn.classList.remove('recording');
    appendLog('[Mic] Recording stopped. Processing...');
  }

  async function processAudio(): Promise<void> {
    appendLog('[Whisper] Transcribing audio buffer via WebGPU...');
    const blob = new Blob(audioChunks, { type: 'audio/webm' });
    await blob.arrayBuffer();

    await new Promise((resolve) => setTimeout(resolve, 30));

    const text = 'Hello, can you explain what WebGPU is?';
    appendLog(`[User] "${text}"`);

    await runLLM(text);
  }

  async function runLLM(prompt: string): Promise<void> {
    appendLog(`[LLM] Generating response for: "${prompt}"...`);

    const responseTokens = ['WebGPU', 'is', 'a', 'modern', 'graphics', 'API'];

    appendLog('[Assistant] ');

    for (const token of responseTokens) {
      await new Promise((resolve) => setTimeout(resolve, 15));
      validLogEl.textContent += `${token} `;
      validLogEl.scrollTop = validLogEl.scrollHeight;
    }
    appendLog(`
[System] Generation complete.`);
  }

  initModels();
}

initWhisperLlmDemo();
