import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { handleZooCommand } from '../src/commands/zoo.js';

describe('handleZooCommand', () => {
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
    await handleZooCommand([]);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Usage: onnx9000 zoo'));
    expect(processExitSpy).toHaveBeenCalledWith(0);

    await handleZooCommand(['-h']);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Usage: onnx9000 zoo'));
    expect(processExitSpy).toHaveBeenCalledWith(0);
  });

  it('should run pull command', async () => {
    await handleZooCommand(['pull', 'HuggingFaceTB/SmolLM-135M']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Executing Zoo command: pull');
    expect(consoleLogSpy).toHaveBeenCalledWith('Downloading HuggingFaceTB/SmolLM-135M...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Zoo subsystem loaded.');
  });

  it('should exit if pull is missing model_id', async () => {
    await handleZooCommand(['pull']);
    expect(consoleErrorSpy).toHaveBeenCalledWith('Usage: onnx9000 zoo pull <model_id>');
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });

  it('should exit on unknown command', async () => {
    await handleZooCommand(['unknown']);
    expect(consoleErrorSpy).toHaveBeenCalledWith('Unknown zoo command: unknown');
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });
});
