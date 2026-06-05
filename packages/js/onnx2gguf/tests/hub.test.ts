import { describe, it, expect, vi } from 'vitest';
import { fetchHfConfig, generateReadme } from '../src/hub.js';

global.fetch = vi.fn().mockResolvedValue({
  ok: true,
  json: async () => ({}),
  text: async () => 'tok',
});

describe('hub', () => {
  it('should fetch hf config', async () => {
    const res = await fetchHfConfig('test');
    expect(res.tokenizer).toBe('tok');
  });

  it('should generate readme', () => {
    const rm = generateReadme('m', 'r', 'q');
    expect(rm).toContain('m GGUF');
  });
});
