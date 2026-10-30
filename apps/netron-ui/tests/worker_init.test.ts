import { beforeAll, describe, expect, it } from 'vitest';

const g = globalThis as typeof globalThis & {
  self: {
    postMessage: () => void;
    onmessage?: (ev: MessageEvent) => Promise<void>;
  };
};

describe('Worker Initialization', () => {
  beforeAll(() => {
    g.self = { postMessage: () => undefined };
  });

  it('should map self.onmessage (lines 31-33)', async () => {
    const _w = await import('../src/parser/worker');
    expect(typeof g.self.onmessage).toBe('function');

    let posted = false;
    const orig = g.self.postMessage;
    g.self.postMessage = () => {
      posted = true;
    };
    await g.self.onmessage?.({} as MessageEvent);
    expect(posted).toBe(true);
    g.self.postMessage = orig;
  });
});
