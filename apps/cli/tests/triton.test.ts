import { describe, it, expect, vi } from 'vitest';
import { handleTritonCommand } from '../src/commands/triton.js';

describe('handleTritonCommand', () => {
  it('should print help and exit if no args or -h', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);

    await handleTritonCommand([]);
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
    expect(exitSpy).toHaveBeenCalledWith(0);

    await handleTritonCommand(['-h']);
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
    expect(exitSpy).toHaveBeenCalledWith(0);

    consoleSpy.mockRestore();
    exitSpy.mockRestore();
  });

  it('should run triton codegen', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleTritonCommand(['dummy.onnx']);
    expect(consoleSpy).toHaveBeenCalledWith('Generating Triton code from dummy.onnx...');
    expect(consoleSpy).toHaveBeenCalledWith('@triton.jit');
    consoleSpy.mockRestore();
  });

  it('should run triton codegen empty model path', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleTritonCommand([undefined as any]);
    expect(consoleSpy).toHaveBeenCalledWith('Generating Triton code from ...');
    expect(consoleSpy).toHaveBeenCalledWith('@triton.jit');
    consoleSpy.mockRestore();
  });
});
