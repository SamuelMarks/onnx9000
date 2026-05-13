import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { handleOptimumCommand } from '../src/commands/optimum.js';

describe('handleOptimumCommand', () => {
  let consoleLogSpy: any;
  let consoleErrorSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should print help and exit if no args or -h', async () => {
    await handleOptimumCommand([]);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Usage: onnx9000 optimum'));
    expect(processExitSpy).toHaveBeenCalledWith(0);
  });

  it('should exit on unknown command', async () => {
    await handleOptimumCommand(['unknown']);
    expect(consoleErrorSpy).toHaveBeenCalledWith('Unknown optimum command: unknown');
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });

  it('should handle export command', async () => {
    await handleOptimumCommand(['export', 'model_id', '--task', 'text-classification']);
    expect(consoleLogSpy).toHaveBeenCalledWith(
      'Exporting model_id for task text-classification...',
    );

    await handleOptimumCommand(['export']);
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'Usage: onnx9000 optimum export <model_id> [options]',
    );
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });

  it('should handle optimize command', async () => {
    await handleOptimumCommand(['optimize', 'model.onnx', '--level', '2', '--optimize-size']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Optimizing model.onnx at level 2 for size...');

    await handleOptimumCommand(['optimize']);
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'Usage: onnx9000 optimum optimize <model> [options]',
    );
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });

  it('should handle quantize command', async () => {
    await handleOptimumCommand(['quantize', 'model.onnx', '--quantize', 'gptq']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Quantizing model.onnx with method gptq...');

    await handleOptimumCommand(['quantize']);
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'Usage: onnx9000 optimum quantize <model> [options]',
    );
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });
});
