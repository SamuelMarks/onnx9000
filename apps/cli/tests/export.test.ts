import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { handleExportCommand } from '../src/commands/export.js';

describe('handleExportCommand', () => {
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
    await handleExportCommand([]);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Usage: onnx9000 export'));
    expect(processExitSpy).toHaveBeenCalledWith(0);

    await handleExportCommand(['-h']);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Usage: onnx9000 export'));
    expect(processExitSpy).toHaveBeenCalledWith(0);
  });

  it('should run export command for format C', async () => {
    await handleExportCommand(['model.onnx', '--format', 'c']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Loading model model.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Transpiling ONNX to C99...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Saving C source to model.c...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Export complete.');
  });

  it('should run export command with custom output', async () => {
    await handleExportCommand(['model.onnx', '--format', 'c', '-o', 'out.c']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Loading model model.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Saving C source to out.c...');
  });

  it('should exit on unsupported format', async () => {
    await handleExportCommand(['model.onnx', '--format', 'unknown']);
    expect(consoleErrorSpy).toHaveBeenCalledWith('Unsupported format: unknown');
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });
});
