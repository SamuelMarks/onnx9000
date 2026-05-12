import { describe, it, expect, vi } from 'vitest';
import { handleCoreMLCommand } from '../src/commands/coreml.js';

describe('handleCoreMLCommand', () => {
  it('should print help and exit if no args or -h', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);

    await handleCoreMLCommand([]);
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
    expect(exitSpy).toHaveBeenCalledWith(0);

    await handleCoreMLCommand(['-h']);
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
    expect(exitSpy).toHaveBeenCalledWith(0);

    consoleSpy.mockRestore();
    exitSpy.mockRestore();
  });

  it('should run coreml export', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleCoreMLCommand(['dummy.onnx']);
    expect(consoleSpy).toHaveBeenCalledWith('Exporting ONNX model to CoreML/MIL: dummy.onnx...');
    expect(consoleSpy).toHaveBeenCalledWith('CoreML AST generated.');
    consoleSpy.mockRestore();
  });

  it('should run coreml export empty model path', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleCoreMLCommand([undefined as any]);
    expect(consoleSpy).toHaveBeenCalledWith('Exporting ONNX model to CoreML/MIL: ...');
    expect(consoleSpy).toHaveBeenCalledWith('CoreML AST generated.');
    consoleSpy.mockRestore();
  });
});
