import { describe, expect, it, vi } from 'vitest';
import { fetchAndParseModel } from '../src/parser/fetcher.js';

vi.mock('@onnx9000/core', () => ({
  parseModelProto: vi.fn().mockReturnValue({ name: 'mock' }),
  BlobReader: class {},
}));

describe('netron-ui fetcher', () => {
  it('should fetch and parse', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-length': '10' }),
      body: {
        getReader: () => {
          let readOnce = false;
          return {
            read: async () => {
              if (!readOnce) {
                readOnce = true;
                return { done: false, value: new Uint8Array([1, 2, 3]) };
              }
              return { done: true };
            },
          };
        },
      },
    });

    const graph = await fetchAndParseModel('https://example.com/model.onnx');
    expect(graph.name).toBe('mock');
  });
});
