import { describe, it, expect, vi } from 'vitest';

describe('debug.js side-effect', () => {
  it('should cover the side-effect when DEBUG_FORCE_RUN is true', async () => {
    // Set the env var BEFORE importing the module
    process.env.DEBUG_FORCE_RUN = 'true';
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    // Import the module
    const debug = await import('../debug.js?test=' + Date.now());

    // Manually call main to ensure it finishes and we can check the spy
    // The top-level call in the file might still be running or finished,
    // but calling it here with await guarantees we see the result.
    await debug.main();

    expect(logSpy).toHaveBeenCalled();

    logSpy.mockRestore();
    delete process.env.DEBUG_FORCE_RUN;
  });
});
