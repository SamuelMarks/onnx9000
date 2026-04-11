import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { resolve } from 'node:path';

describe('CLI', () => {
  let originalArgv: string[];
  let exitMock: Object;
  let errorMock: Object;
  let logMock: Object;

  beforeEach(() => {
    originalArgv = process.argv;
    exitMock = vi.spyOn(process, 'exit').mockImplementation(() => undefined);
    errorMock = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    logMock = vi.spyOn(console, 'log').mockImplementation(() => undefined);
    vi.resetModules();
  });

  afterEach(() => {
    process.argv = originalArgv;
    vi.restoreAllMocks();
  });

  const runCLI = async (args: string[]) => {
    process.argv = ['node', 'cli.js', ...args];
    try {
      await import('../src/cli.js' + '?t=' + Date.now());
      // wait a tick for promises
      await new Promise((r) => setTimeout(r, 10));
    } catch (e) {
      if (e.message !== 'exit') throw e;
    }
  };

  it('should run export command', async () => {
    await runCLI(['export', 'dummy.onnx']);
    expect(logMock).toHaveBeenCalledWith(expect.stringContaining('Loading ONNX model from'));
    expect(logMock).toHaveBeenCalledWith(
      expect.stringContaining('Converting to CoreML MIL Program...'),
    );
  });

  it('should run import command', async () => {
    await runCLI(['import', 'dummy.mlpackage']);
    expect(logMock).toHaveBeenCalledWith(expect.stringContaining('Importing CoreML package from'));
  });

  it('should fail with missing args', async () => {
    await runCLI(['export']);
    expect(errorMock).toHaveBeenCalledWith('Usage: onnx9000-coreml <export|import> <model_path>');
    expect(exitMock).toHaveBeenCalledWith(1);
  });

  it('should fail with unknown command', async () => {
    await runCLI(['foo', 'bar']);
    expect(errorMock).toHaveBeenCalledWith('Unknown command: foo');
    expect(exitMock).toHaveBeenCalledWith(1);
  });
});
