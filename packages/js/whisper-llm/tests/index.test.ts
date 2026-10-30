import { describe, expect, it } from 'vitest';
import { WhisperLlm } from '../src/index.js';

describe('WhisperLlm', () => {
  it('should transcribe audio strings and extract log-energy frames', () => {
    const w = new WhisperLlm();
    expect(w.transcribe('sample.wav')).toBe('[Whisper-LLM] transcribed sample.wav');
    expect(() => w.transcribe('')).toThrow('Invalid audio string');

    const energies = w.extractLogEnergy(new Float32Array([0.5, -0.5, 0.25, 0.25]));
    expect(energies).toHaveLength(1);
    expect(() => w.extractLogEnergy(new Float32Array([0.1]), 0)).toThrow(
      'frameSize must be positive',
    );
  });
});
