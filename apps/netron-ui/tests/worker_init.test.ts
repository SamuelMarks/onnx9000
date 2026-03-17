import { describe, it, expect, beforeAll } from 'vitest';

describe('Worker Initialization', () => {
  beforeAll(() => {
    (globalThis as any).self = { postMessage: () => {} };
  });

  it('should map self.onmessage (lines 31-33)', async () => {
    const w = await import('../src/parser/worker');
    expect(typeof (globalThis as any).self.onmessage).toBe('function');

    let posted = false;
    const orig = (globalThis as any).self.postMessage;
    (globalThis as any).self.postMessage = () => {
      posted = true;
    };
    await (globalThis as any).self.onmessage({} as any);
    expect(posted).toBe(true);
    (globalThis as any).self.postMessage = orig;
  });
});
