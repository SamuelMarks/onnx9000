import { describe, it, expect } from 'vitest';
import { whisperTiny } from '../src/models/whisper.js';

describe('Whisper', () => {
  it('should create and call', () => {
    const model = whisperTiny();
    expect(model).toBeDefined();
    const out = model.call({} as any, {} as any);
    expect(out).toBeDefined();
  });
});
