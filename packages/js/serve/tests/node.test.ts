import { describe, expect, it, vi } from 'vitest';
import { serveNode } from '../src/node.js';

vi.mock('node:http', () => ({
  createServer: vi.fn().mockReturnValue({ listen: vi.fn() }),
}));

describe('node server', () => {
  it('should serve node', () => {
    const srv = serveNode({} as never as Parameters<typeof serveNode>[0], 8080);
    expect(srv).toBeDefined();
  });
});
