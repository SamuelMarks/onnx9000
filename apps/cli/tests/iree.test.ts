import { describe, it, expect, vi } from 'vitest';
import { handleIreeCommand } from '../src/commands/iree.js';
import * as compiler from '@onnx9000/iree-compiler/src/cli.js';

vi.mock('@onnx9000/iree-compiler/src/cli.js', () => ({
  compileModel: vi.fn().mockResolvedValue(undefined),
}));

describe('handleIreeCommand', () => {
  it('should print help and exit if no args or -h', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);

    await handleIreeCommand([]);
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
    expect(exitSpy).toHaveBeenCalledWith(0);

    await handleIreeCommand(['-h']);
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
    expect(exitSpy).toHaveBeenCalledWith(0);

    consoleSpy.mockRestore();
    exitSpy.mockRestore();
  });

  it('should compile model', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleIreeCommand(['compile', 'dummy.onnx']);
    expect(compiler.compileModel).toHaveBeenCalledWith('dummy.onnx', expect.any(Object));
    consoleSpy.mockRestore();
  });

  it('should compile model fallback args', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleIreeCommand(['compile']);
    expect(compiler.compileModel).toHaveBeenCalledWith('', expect.any(Object));
    consoleSpy.mockRestore();
  });

  it('should run model', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleIreeCommand(['run', 'dummy.wvm']);
    expect(consoleSpy).toHaveBeenCalledWith('Running dummy.wvm via IREE WVM...');
    expect(consoleSpy).toHaveBeenCalledWith('Execution successful.');
    consoleSpy.mockRestore();
  });

  it('should handle unknown command', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleIreeCommand(['unknown']);
    expect(consoleSpy).toHaveBeenCalledWith('Invalid IREE command. Use compile or run.');
    consoleSpy.mockRestore();
  });
});

it('should run model fallback args', async () => {
  const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  await handleIreeCommand(['run']);
  expect(consoleSpy).toHaveBeenCalledWith('Running  via IREE WVM...');
  expect(consoleSpy).toHaveBeenCalledWith('Execution successful.');
  consoleSpy.mockRestore();
});
