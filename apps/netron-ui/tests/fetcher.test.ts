import { describe, it, expect, vi } from 'vitest';
import { fetchAndParseModel } from '../src/parser/fetcher';
import * as core from '@onnx9000/core';

describe('Fetcher', () => {
  it('should translate github blob url', async () => {
    // Mock fetch
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-length': '10' }),
      body: {
        getReader: () => ({
          read: vi
            .fn()
            .mockResolvedValueOnce({ done: false, value: new Uint8Array([1]) })
            .mockResolvedValueOnce({ done: true }),
        }),
      },
    });

    vi.spyOn(core, 'parseModelProto').mockResolvedValue('Graph' as any);

    let prog = 0;
    const cb = (p: number) => {
      prog = p;
    };
    const g = await fetchAndParseModel('https://github.com/user/repo/blob/master/model.onnx', cb);
    expect(g).toBe('Graph');
    expect(global.fetch).toHaveBeenCalledWith(
      'https://raw.githubusercontent.com/user/repo/master/model.onnx',
    );
    expect(prog).toBe(10);
  });

  it('should throw on fetch error', async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: false, statusText: 'Not Found' });
    await expect(fetchAndParseModel('http://example.com/model')).rejects.toThrow(
      'Failed to fetch model from http://example.com/model: Not Found',
    );
  });

  it('should throw on missing body', async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true, headers: new Headers() });
    await expect(fetchAndParseModel('http://example.com/model')).rejects.toThrow(
      'ReadableStream not supported by browser.',
    );
  });
});
