import { describe, it, expect } from 'vitest';
import { WhisperLlm } from '../src/index';

describe('WhisperLlm', () => {
  it('should transcribe audio', () => {
    const whisper = new WhisperLlm();
    expect(whisper.transcribe('audio_data')).toBe('[Whisper-LLM] transcribed audio_data');
  });

  it('should throw on empty string', () => {
    const whisper = new WhisperLlm();
    expect(() => whisper.transcribe('')).toThrow('Invalid audio string');
  });
});
