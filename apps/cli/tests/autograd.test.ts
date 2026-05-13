import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { handleAutogradCommand } from '../src/commands/autograd.js';

describe('handleAutogradCommand', () => {
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
    await handleAutogradCommand([]);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Usage: onnx9000 autograd'));
    expect(processExitSpy).toHaveBeenCalledWith(0);

    await handleAutogradCommand(['-h']);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Usage: onnx9000 autograd'));
    expect(processExitSpy).toHaveBeenCalledWith(0);
  });

  it('should run autograd command with default output', async () => {
    await handleAutogradCommand(['dummy.onnx']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Loading forward graph dummy.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Generating backward graph...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Saving backward graph to dummy_bw.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Autograd complete.');
  });

  it('should run autograd command with custom output', async () => {
    await handleAutogradCommand(['dummy.onnx', '-o', 'custom_bw.onnx']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Loading forward graph dummy.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Generating backward graph...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Saving backward graph to custom_bw.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Autograd complete.');
  });
});
