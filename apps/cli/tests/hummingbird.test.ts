import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { handleHummingbirdCommand } from '../src/commands/hummingbird.js';

describe('handleHummingbirdCommand', () => {
  let consoleLogSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should print help and exit if no args or -h', async () => {
    await handleHummingbirdCommand([]);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining('Usage: onnx9000 hummingbird'),
    );
    expect(processExitSpy).toHaveBeenCalledWith(0);

    await handleHummingbirdCommand(['-h']);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining('Usage: onnx9000 hummingbird'),
    );
    expect(processExitSpy).toHaveBeenCalledWith(0);
  });

  it('should run hummingbird command with default output', async () => {
    await handleHummingbirdCommand(['model.onnx']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Loading tree model model.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Transpiling to tensor operations...');
    expect(consoleLogSpy).toHaveBeenCalledWith(
      'Saving optimized tensor model to model_tensor.onnx...',
    );
    expect(consoleLogSpy).toHaveBeenCalledWith('Hummingbird conversion complete.');
  });

  it('should run hummingbird command with custom output', async () => {
    await handleHummingbirdCommand(['model.onnx', '-o', 'custom_tensor.onnx']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Loading tree model model.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Transpiling to tensor operations...');
    expect(consoleLogSpy).toHaveBeenCalledWith(
      'Saving optimized tensor model to custom_tensor.onnx...',
    );
    expect(consoleLogSpy).toHaveBeenCalledWith('Hummingbird conversion complete.');
  });
});
