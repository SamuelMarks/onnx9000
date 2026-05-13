import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { handleSparseCommand } from '../src/commands/sparse.js';

describe('handleSparseCommand', () => {
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
    await handleSparseCommand([]);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Usage: onnx9000 sparse'));
    expect(processExitSpy).toHaveBeenCalledWith(0);

    await handleSparseCommand(['-h']);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Usage: onnx9000 sparse'));
    expect(processExitSpy).toHaveBeenCalledWith(0);
  });

  it('should exit on unknown command', async () => {
    await handleSparseCommand(['unknown']);
    expect(consoleErrorSpy).toHaveBeenCalledWith('Unknown sparse command: unknown');
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });

  it('should exit if prune missing model', async () => {
    await handleSparseCommand(['prune']);
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'Usage: onnx9000 sparse prune <model.onnx> [options]',
    );
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });

  it('should run prune command with default output', async () => {
    await handleSparseCommand(['prune', 'model.onnx']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Loading model model.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Pruning model to 0.0 sparsity...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Saving sparse model to model_sparse.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Sparsification complete.');
  });

  it('should run prune command with custom output and sparsity', async () => {
    await handleSparseCommand(['prune', 'model.onnx', '--sparsity', '0.8', '-o', 'out.onnx']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Loading model model.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Pruning model to 0.8 sparsity...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Saving sparse model to out.onnx...');
  });

  it('should run prune command with recipe', async () => {
    await handleSparseCommand(['prune', 'model.onnx', '--recipe', 'recipe.yaml']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Loading model model.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Applying pruning recipe: recipe.yaml');
    expect(consoleLogSpy).toHaveBeenCalledWith('Saving sparse model to model_sparse.onnx...');
  });
});
