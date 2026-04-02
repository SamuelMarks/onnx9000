import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('Triton Worker', () => {
  it('should handle messages and generate triton code', async () => {
    // Mock self.onmessage and postMessage
    const originalSelf = global.self;
    const postMessageSpy = vi.fn();
    (global as any).self = {
      postMessage: postMessageSpy,
    };

    // Import worker to register listener
    await import('../src/worker.js?t=' + Date.now());

    const onmessage = (global as any).self.onmessage;
    expect(onmessage).toBeDefined();

    // Trigger message with a named graph
    const mockGraph = { name: 'test_graph', nodes: [], inputs: [], outputs: [] };
    onmessage({ data: { graph: mockGraph, config: {} } });

    expect(postMessageSpy).toHaveBeenCalled();
    const result = postMessageSpy.mock.calls[0][0];
    expect(result.code).toBeDefined();

    // Cleanup
    (global as any).self = originalSelf;
  });
});
