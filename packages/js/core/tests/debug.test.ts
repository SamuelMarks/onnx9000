import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Graph, Node } from '../src/index.ts';

vi.mock('../src/index.ts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/index.ts')>();
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
    const g = new Graph('test');
    g.nodes.push(new Node('Relu'));
    vi.mocked(load).mockResolvedValue(g);

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
