import { describe, expect, it, vi } from 'vitest';
import { serveNode } from '../src/node.js';

let capturedHandler: (req: unknown, res: unknown) => Promise<void>;
vi.mock('node:http2', () => ({
  createServer: (handler: (req: unknown, res: unknown) => Promise<void>) => {
    capturedHandler = handler;
    return { listen: vi.fn() };
  },
}));

describe('Node Serve HTTP/2 Mock', () => {
  it('handles HTTP/2 paths', async () => {
    const mockServer = {
      fetch: vi.fn().mockResolvedValue({
        status: 200,
        headers: new Headers({ 'x-test': '1' }),
        body: null,
      }),
    };

    serveNode(mockServer as never as Parameters<typeof serveNode>[0], 0, true);
    expect(capturedHandler).toBeDefined();

    const mockReq = {
      socket: { encrypted: true },
      headers: {
        ':authority': 'localhost',
        ':method': 'POST',
        'array-head': ['a', 'b'],
      },
      url: '/test',
      [Symbol.asyncIterator]: async function* () {
        yield Buffer.from('hello');
      },
    };

    const mockRes = {
      stream: {
        respond: vi.fn(),
      },
      end: vi.fn(),
    };

    await capturedHandler(mockReq, mockRes);
    expect(mockRes.stream.respond).toHaveBeenCalled();

    // Test the crash path for http2
    const crashServer = {
      fetch: vi.fn().mockRejectedValue(new Error('crash')),
    };
    serveNode(crashServer as never as Parameters<typeof serveNode>[0], 0, true);
    await capturedHandler(mockReq, mockRes);
    expect(mockRes.stream.respond).toHaveBeenCalledWith({ ':status': 500 });
  });
});
