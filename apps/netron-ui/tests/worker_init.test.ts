import { describe, it, expect, beforeAll } from 'vitest';

describe('Worker Initialization', () => {
  beforeAll(() => {
    (globalThis as Object).self = { postMessage: () => {} };
  });

  it('should map self.onmessage (lines 31-33)', async () => {
    const w = await import('../src/parser/worker');
    expect(typeof (globalThis as Object).self.onmessage).toBe('function');

    let posted = false;
    const orig = (globalThis as Object).self.postMessage;
    (globalThis as Object).self.postMessage = () => {
      posted = true;
    };
    await (globalThis as Object).self.onmessage({} as Object);
    expect(posted).toBe(true);
    (globalThis as Object).self.postMessage = orig;
  });
});
