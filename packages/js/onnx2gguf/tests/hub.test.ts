import { expect, test, vi } from 'vitest';
import { fetchHfConfig, generateReadme } from '../src/hub';

test('generateReadme', () => {
  const readme = generateReadme('Llama-2-7b', 'meta-llama/Llama-2-7b', 'Q4_0');
  expect(readme).toContain('base_model: meta-llama/Llama-2-7b');
  expect(readme).toContain('- **Level:** Q4_0');
});

test('fetchHfConfig 404', async () => {
  global.fetch = vi.fn().mockResolvedValue({
    ok: false,
    status: 404,
  });

  const res = await fetchHfConfig('dummy/repo');
  expect(res.config).toEqual({});
  expect(res.tokenizer).toBe('');
  expect(res.url).toBe('https://huggingface.co/dummy/repo');
});

test('fetchHfConfig success', async () => {
  global.fetch = vi.fn().mockImplementation((url: string, opts: Object) => {
    if (url.endsWith('config.json'))
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ architectues: ['LlamaForCausalLM'] }),
      });
    if (url.endsWith('tokenizer.json'))
      return Promise.resolve({ ok: true, text: () => Promise.resolve('{"vocab": {}}') });
    return Promise.resolve({ ok: false });
  });

  const res = await fetchHfConfig('dummy/repo', 'token123');
  expect(res.config).toHaveProperty('architectues');
  expect(res.tokenizer).toContain('vocab');
  expect(global.fetch).toHaveBeenCalledWith(
    expect.any(String),
    expect.objectContaining({
      headers: { Authorization: 'Bearer token123' },
    }),
  );
});
