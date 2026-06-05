import { describe, it, expect, vi } from 'vitest';
import { initWhisperLlmDemo } from '../src/main.js';

describe('demo-whisper-llm main', () => {
  it('should run main demo', async () => {
    vi.useFakeTimers();
    document.body.innerHTML = `
      <div id="log"></div>
      <button id="record-btn"></button>
      <button id="clear-btn"></button>
    `;

    // Mock getUserMedia and MediaRecorder
    (global.navigator as any).mediaDevices = {
      getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: vi.fn() }] }),
    };
    (global as any).MediaRecorder = class {
      state = 'inactive';
      stream = { getTracks: () => [{ stop: vi.fn() }] };
      start() {
        this.state = 'recording';
      }
      stop() {
        this.state = 'inactive';
        if (this.onstop) this.onstop();
      }
      ondataavailable = null;
      onstop = null;
    };

    initWhisperLlmDemo();
    await new Promise((r) => process.nextTick(r)); // allow initModels to log

    document.getElementById('record-btn')?.click();
    await new Promise((r) => process.nextTick(r)); // allow getUserMedia
    expect(document.getElementById('log')?.textContent).toContain('Recording started');

    document.getElementById('record-btn')?.click(); // stop recording

    // fast forward transcription and llm streaming
    for (let i = 0; i < 30; i++) {
      vi.runAllTimers();
      await new Promise((r) => process.nextTick(r));
    }

    expect(document.getElementById('log')?.textContent).toContain('Generation complete');

    document.getElementById('clear-btn')?.click();
    expect(document.getElementById('log')?.textContent).toContain('Log cleared');
    vi.useRealTimers();
  });
});
