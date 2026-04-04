import { describe, it, expect, vi } from 'vitest';
import { serveNode } from '../src/node';
import * as http from 'node:http';

describe('Node Serve', () => {
  it('handles GET request', async () => {
    const mockServer = {
      fetch: vi.fn().mockResolvedValue({
        status: 200,
        headers: new Headers({ 'x-test': '1', 'content-type': 'text/plain' }),
        body: null,
      }),
    };

    const server = serveNode(mockServer as any, 0, false);
    await new Promise((resolve) => server.on('listening', resolve));

    const port = (server.address() as any).port;

    const res = await new Promise<http.IncomingMessage>((resolve) => {
      http.get(`http://localhost:${port}/test`, resolve);
    });

    expect(res.statusCode).toBe(200);
    expect(res.headers['x-test']).toBe('1');

    server.close();
  });

  it('handles POST request with body and stream response', async () => {
    const mockServer = {
      fetch: vi.fn().mockImplementation(async (req) => {
        const bodyText = await req.text();
        return {
          status: 200,
          headers: new Headers({ 'x-echo': bodyText }),
          body: {
            getReader: () => {
              let count = 0;
              return {
                read: async () =>
                  count++ === 0
                    ? { done: false, value: new TextEncoder().encode('streamed') }
                    : { done: true },
              };
            },
          },
        };
      }),
    };

    const server = serveNode(mockServer as any, 0, false);
    await new Promise((resolve) => server.on('listening', resolve));

    const port = (server.address() as any).port;

    const res = await new Promise<http.IncomingMessage>((resolve) => {
      const req = http.request(
        `http://localhost:${port}/`,
        { method: 'POST', headers: { 'multi-val': ['a', 'b'] } },
        resolve,
      );
      req.write('hello-post');
      req.end();
    });

    expect(res.statusCode).toBe(200);
    expect(res.headers['x-echo']).toBe('hello-post');

    const chunks = [];
    for await (const chunk of res) {
      chunks.push(chunk);
    }
    expect(Buffer.concat(chunks).toString()).toBe('streamed');

    server.close();
  });

  it('handles fetch error natively', async () => {
    const mockServer = {
      fetch: vi.fn().mockRejectedValue(new Error('crash')),
    };

    const server = serveNode(mockServer as any, 0, false);
    await new Promise((resolve) => server.on('listening', resolve));

    const port = (server.address() as any).port;

    const res = await new Promise<http.IncomingMessage>((resolve) => {
      http.get(`http://localhost:${port}/test`, resolve);
    });

    expect(res.statusCode).toBe(500);
    const chunks = [];
    for await (const chunk of res) {
      chunks.push(chunk);
    }
    expect(Buffer.concat(chunks).toString()).toContain('crash');

    server.close();
  });
});
