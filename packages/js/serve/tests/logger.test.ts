import { describe, expect, it, vi } from 'vitest';
import { extractTraceContext, Logger, LogLevel } from '../src/logger.js';

describe('logger', () => {
  it('should log', () => {
    const l = new Logger(LogLevel.DEBUG);
    const spy = vi.spyOn(console, 'debug').mockImplementation(() => {});
    l.debug('test');
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it('should extract trace', () => {
    const req = new Request('http://localhost', {
      headers: { traceparent: '123' },
    });
    const ctx = extractTraceContext(req);
    expect(ctx.traceparent).toBe('123');
  });
});
