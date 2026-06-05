import { describe, it, expect, vi } from 'vitest';
import { initGenAIDemo } from '../src/main.js';

describe('demo-genai', () => {
  it('should initialize and handle click', () => {
    vi.useFakeTimers();
    document.body.innerHTML = '<button id="run-btn"></button><div id="output"></div>';
    initGenAIDemo();
    document.getElementById('run-btn')?.click();
    expect(document.getElementById('output')?.innerText).toBe('Initializing GenAI Subsystem...');
    vi.runAllTimers();
    expect(document.getElementById('output')?.innerText).toBe(
      'GenAI models loaded.\nExecution complete: SUCCESS',
    );
    vi.useRealTimers();
  });
});
