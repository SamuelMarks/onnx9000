import { describe, it, expect, vi } from 'vitest';
import { readBrowserFile, fetchRemoteUrl } from '../../src/keras/browser-io.js';

describe('browser-io', () => {
  it('should read file', async () => {
    const file = new File(['123'], 'test');
    const res = await readBrowserFile(file);
    expect(res.byteLength).toBe(3);
  });

  it('should fetch remote', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new Uint8Array([1, 2]).buffer,
    });

    const res = await fetchRemoteUrl('http://test');
    expect(res.byteLength).toBe(2);
  });
});
