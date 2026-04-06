import { describe, it, expect, vi } from 'vitest';
import * as index from '../src/index.ts';
import { run } from '../debug.js';

describe('debug.js', () => {
  it('should run debug script', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    // Mock load to return a dummy graph
    vi.spyOn(index, 'load').mockResolvedValue({ nodes: [] } as Object);

    process.env.DEBUG_FORCE_RUN = 'true';
    await run();
    expect(logSpy).toHaveBeenCalledWith('Success! Nodes:', 0);
  });

  it('should catch errors in debug script', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(index, 'load').mockRejectedValue(new Error('fail'));

    process.env.DEBUG_FORCE_RUN = 'true';
    await run();
    expect(errorSpy).toHaveBeenCalled();
  });
});
