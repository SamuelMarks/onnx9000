import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as onnx from '../src/parser/onnx.js';

vi.mock('../src/parser/onnx.js', async (importOriginal) => {
  const actual = (await importOriginal()) as any;
  return {
    ...actual,
    parseModelProto: vi.fn(),
  };
});

describe('debug script', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
  });

  it('should run debug script successfully', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    const { parseModelProto } = await import('../src/parser/onnx.js');
    vi.mocked(parseModelProto).mockResolvedValue({ nodes: { length: 1 } } as any);

    // Import run from debug script and call it manually
    const { run } = await import('../debug.js');
    await run();

    expect(logSpy).toHaveBeenCalledWith('Success! Nodes:', 1);

    logSpy.mockRestore();
    errorSpy.mockRestore();
  });

  it('should handle errors in debug script', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    const { parseModelProto } = await import('../src/parser/onnx.js');
    const testError = new Error('test error');
    vi.mocked(parseModelProto).mockRejectedValue(testError);

    const { run } = await import('../debug.js');
    await run();

    expect(errorSpy).toHaveBeenCalledWith(testError);

    errorSpy.mockRestore();
  });

  it('should run automatically if not in test env', async () => {
    vi.stubGlobal('process', { env: { NODE_ENV: 'development' } });
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    vi.resetModules();
    await import('../debug.js');

    // Wait for the async run() to execute
    await new Promise((resolve) => setTimeout(resolve, 100));

    expect(logSpy).toHaveBeenCalled();

    logSpy.mockRestore();
    vi.unstubAllGlobals();
  });
});
