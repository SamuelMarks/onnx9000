import { describe, it, expect, vi } from 'vitest';
import { HashRing, PeerRegistry, proxyRequest } from '../src/routing.js';

describe('routing', () => {
  it('should hash ring', () => {
    const r = new HashRing();
    r.addNode('A');
    r.addNode('B');
    expect(r.getNode('test')).toBeDefined();
    r.removeNode('A');
  });

  it('should registry', () => {
    const r = new PeerRegistry();
    r.register('m1', 'A');
    expect(r.getNextNodeForModel('m1')).toBe('A');
  });

  it('should proxy', async () => {
    global.fetch = vi.fn().mockResolvedValue(new Response('ok'));
    const req = new Request('http://localhost/test');
    const res = await proxyRequest(req, 'http://remote');
    expect(await res.text()).toBe('ok');
  });
});
