import { describe, it, expect, vi } from 'vitest';
import { HashRing, PeerRegistry, proxyRequest } from '../src/routing';
import { globalLogger, LogLevel } from '../src/logger';
import { DynamicBatcher } from '../src/batcher';
import Onnx9000ServerInstance, { createServer } from '../src/index';

describe('Coverage Routing, Index, Logger, Batcher', () => {
  it('Routing - HashRing', () => {
    const ring = new HashRing();
    expect(ring.getNode('key')).toBeNull();

    (ring as any).addNode('node2');
    (ring as any).addNode('node1');
    (ring as any).addNode('node2'); // duplicate

    expect(ring.getNode('key')).toBeDefined();

    (ring as any).removeNode('node2');
    expect(ring.getNode('key')).toBe('node1');
  });

  it('Routing - PeerRegistry', () => {
    const registry = new PeerRegistry();
    expect(registry.getNextNodeForModel('m1')).toBeNull();

    registry.register('m1', 'nodeA');
    registry.register('m1', 'nodeB');
    registry.register('m1', 'nodeA'); // duplicate

    expect(registry.getNextNodeForModel('m1')).toBe('nodeA');
    expect(registry.getNextNodeForModel('m1')).toBe('nodeB');
    expect(registry.getNextNodeForModel('m1')).toBe('nodeA');
  });

  it('Routing - proxyRequest', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('proxied'));
    vi.stubGlobal('fetch', fetchMock);

    const req1 = new Request('http://localhost/path', {
      method: 'GET',
      headers: { 'cf-connecting-ip': '1.1.1.1' },
    });
    const res1 = await proxyRequest(req1, 'https://backend:8080');
    expect(await res1.text()).toBe('proxied');
    expect(fetchMock.mock.calls[0][0]).toBe('https://backend:8080/path');

    const req2 = new Request('http://localhost/post', {
      method: 'POST',
      body: 'body',
      headers: { 'x-forwarded-for': '2.2.2.2' },
    });
    const res2 = await proxyRequest(req2, 'http://backend2');
    expect(fetchMock.mock.calls[1][1].body).toBeDefined();

    vi.unstubAllGlobals();
  });

  it('Logger - all exports', async () => {
    globalLogger.exporterUrl = 'http://test';
    const fetchMock = vi.fn().mockResolvedValue(new Response());
    vi.stubGlobal('fetch', fetchMock);
    globalLogger.level = LogLevel.TRACE;
    globalLogger.debug('debug');
    globalLogger.info('info');
    globalLogger.warn('warn');
    globalLogger.error('error');
    expect(fetchMock).toHaveBeenCalledTimes(4);
    vi.unstubAllGlobals();
  });

  it('Batcher - timeout', async () => {
    vi.useFakeTimers();
    let executeMock = vi.fn().mockResolvedValue(['res1']);
    const batcher = new DynamicBatcher(executeMock, { maxBatchSize: 5, batchTimeoutMs: 100 });

    const p = batcher.add({ val: 1 });

    vi.advanceTimersByTime(200);
    const res = await p;
    expect(res).toBe('res1');
    expect(executeMock).toHaveBeenCalledTimes(1);

    vi.useRealTimers();
  });

  it('Index - Default Export', async () => {
    const server = createServer();
    expect(server.router).toBeDefined();

    const req = new Request('http://localhost/notfound');
    const res = await Onnx9000ServerInstance.fetch(req);
    expect(res.status).toBe(404);
  });
});
