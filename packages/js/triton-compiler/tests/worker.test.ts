import { describe, it, expect, vi } from 'vitest';

interface MockSelf {
  postMessage: ReturnType<typeof vi.fn>;
  onmessage?: (e: {
    data: {
      graph: { name: string; nodes: string[]; inputs: string[]; outputs: string[] };
      config: Record<string, string>;
    };
  }) => void;
}

interface GlobalWithSelf {
  self?: typeof globalThis | MockSelf;
}

describe('Triton Worker', () => {
  it('should handle messages and generate triton code', async () => {
    // Mock self.onmessage and postMessage
    const globalContext = globalThis as GlobalWithSelf;
    const originalSelf = globalContext.self;
    const postMessageSpy = vi.fn();

    globalContext.self = {
      postMessage: postMessageSpy,
    };

    // Import worker to register listener
    await import('../src/worker.js?t=' + Date.now().toString());

    const onmessage = (globalContext.self as MockSelf).onmessage!;
    expect(onmessage).toBeDefined();

    // Trigger message with a named graph
    const mockGraph = {
      name: 'test_graph',
      nodes: [] as string[],
      inputs: [] as string[],
      outputs: [] as string[],
    };
    onmessage({ data: { graph: mockGraph, config: {} } });

    expect(postMessageSpy).toHaveBeenCalled();
    const result = postMessageSpy.mock.calls[0][0];
    expect(result.code).toBeDefined();

    // Cleanup
    if (originalSelf) {
      globalContext.self = originalSelf;
    } else {
      delete globalContext.self;
    }
  });
});
