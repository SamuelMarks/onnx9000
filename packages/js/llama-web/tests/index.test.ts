import { describe, it, expect } from 'vitest';
import { LlamaWeb } from '../src/index';

describe('LlamaWeb', () => {
  it('should process a llama model', () => {
    const llama = new LlamaWeb();
    expect(llama.run('llama_model')).toBe('[LLaMA-Web] processing llama_model');
  });

  it('should throw on empty string', () => {
    const llama = new LlamaWeb();
    expect(() => llama.run('')).toThrow('Invalid model string');
  });
});
