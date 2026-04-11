import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as index from '../src/index.ts';

vi.mock('../src/index.ts', async (importOriginal) => {
  const actual = (await importOriginal()) as Object;
  return {
    ...actual,
    load: vi.fn(),
  };
});

describe('debug script', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
  });

  it('should run debug script successfully', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);

    const { load } = await import('../src/index.ts');
    vi.mocked(load).mockResolvedValue({ nodes: { length: 1 } } as Object);

    // Import run from debug script and call it manually
    const { run } = await import('../debug.js');
    await run();

    expect(logSpy).toHaveBeenCalledWith('Success! Nodes:', 1);

    logSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it('should handle errors in debug script', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);

    const { load } = await import('../src/index.ts');
    const testError = new Error('test error');
    vi.mocked(load).mockRejectedValue(testError);

    const { run } = await import('../debug.js');
    await run();

    expect(errorSpy).toHaveBeenCalledWith(testError);

    errorSpy.mockRestore();
  });
});
