import { describe, expect, it, vi } from 'vitest';
import { fetchSafetensorsHeader, SafeTensors } from '../src/parser/safetensors.js';

global.fetch = vi.fn().mockResolvedValue({
  ok: true,
  status: 200,
  arrayBuffer: async () => {
    const buf = new Uint8Array(8 + 50);
    const view = new DataView(buf.buffer);
    view.setBigUint64(0, BigInt(19), true);
    const header = '{"__metadata__":{}}';
    buf.set(new TextEncoder().encode(header), 8);
    return buf.buffer;
  },
  headers: new Headers(),
});

describe('safetensors', () => {
  it('should parse', async () => {
    const buf = new Uint8Array(8 + 50);
    const view = new DataView(buf.buffer);
    view.setBigUint64(0, BigInt(19), true);
    const header = '{"__metadata__":{}}';
    buf.set(new TextEncoder().encode(header), 8);

    const st = new SafeTensors(buf.buffer);
    expect(st.metadata).toBeDefined();
  });

  it('should fetch header', async () => {
    const res = await fetchSafetensorsHeader('http://test');
    expect(res).toBeDefined();
  });
});
