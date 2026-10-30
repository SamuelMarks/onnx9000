import { describe, expect, it } from 'vitest';
import { initWhisperLlmIndex } from '../index.js';

describe('demo-whisper-llm index', () => {
  it('should be valid', () => {
    expect(initWhisperLlmIndex).toBeDefined();
  });
});
