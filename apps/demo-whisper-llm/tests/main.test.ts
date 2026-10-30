// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { initWhisperLlmDemo } from '../src/main.js';

describe('demo-whisper-llm', () => {
  beforeEach(() => {
    document.body.innerHTML =
      '<button id="record-btn"></button><button id="clear-btn"></button><div id="log"></div>';
  });

  it('should record, transcribe, generate LLM tokens, clear logs, and handle mic errors', async () => {
    document.body.innerHTML = '';
    initWhisperLlmDemo();

    document.body.innerHTML =
      '<button id="record-btn"></button><button id="clear-btn"></button><div id="log"></div>';
    initWhisperLlmDemo();

    const recordBtn = document.getElementById('record-btn') as HTMLButtonElement;
    const clearBtn = document.getElementById('clear-btn') as HTMLButtonElement;
    const logEl = document.getElementById('log') as HTMLElement;

    clearBtn.click();
    expect(logEl.textContent).toBe('Log cleared.');

    const stopTrackSpy = vi.fn();
    const mockStream = {
      getTracks: () => [{ stop: stopTrackSpy }],
    };

    Object.defineProperty(navigator, 'mediaDevices', {
      value: {
        getUserMedia: vi.fn().mockResolvedValueOnce(mockStream),
      },
      configurable: true,
    });

    class MockMediaRecorder {
      state = 'inactive';
      stream = mockStream;
      ondataavailable?: (e: { data: Blob }) => void;
      onstop?: () => void;
      start(): void {
        this.state = 'recording';
      }
      stop(): void {
        this.state = 'inactive';
        const chunk = new Blob(['audio']);
        chunk.arrayBuffer = async () => new ArrayBuffer(5);
        this.ondataavailable?.({ data: chunk });
        this.onstop?.();
      }
    }
    (globalThis as never as Record<string, unknown>).MediaRecorder = MockMediaRecorder;

    recordBtn.click();
    await new Promise((r) => setTimeout(r, 20));
    expect(logEl.textContent).toContain('[Mic] Recording started...');

    recordBtn.click();
    await new Promise((r) => setTimeout(r, 200));
    expect(stopTrackSpy).toHaveBeenCalled();
    expect(logEl.textContent).toContain('Generation complete.');

    // Inactive recorder & undefined stream branch
    Object.defineProperty(navigator, 'mediaDevices', {
      value: {
        getUserMedia: vi.fn().mockResolvedValueOnce(undefined),
      },
      configurable: true,
    });
    class MockInactiveRecorder {
      state = 'inactive';
      stream = undefined;
      ondataavailable?: (e: { data: Blob }) => void;
      onstop?: () => void;
      start(): void {
        this.state = 'inactive';
      }
      stop(): void {}
    }
    (globalThis as never as Record<string, unknown>).MediaRecorder = MockInactiveRecorder;
    recordBtn.click();
    await new Promise((r) => setTimeout(r, 20));
    recordBtn.click();

    // Error branch
    Object.defineProperty(navigator, 'mediaDevices', {
      value: {
        getUserMedia: vi
          .fn()
          .mockRejectedValueOnce(new Error('Permission denied'))
          .mockRejectedValueOnce('Raw mic error'),
      },
      configurable: true,
    });
    recordBtn.click();
    await new Promise((r) => setTimeout(r, 20));
    expect(logEl.textContent).toContain('Permission denied');

    recordBtn.click();
    await new Promise((r) => setTimeout(r, 20));
    expect(logEl.textContent).toContain('Raw mic error');
  });
});
