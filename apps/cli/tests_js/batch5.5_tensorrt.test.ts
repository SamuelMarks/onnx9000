import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { handleTensorRTCommand } from '../src/commands/tensorrt.js';

describe('CLI Commands Batch 5', () => {
  let consoleLogSpy: any;
  let _consoleErrorSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    _consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation(() => undefined as never);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('handleTensorRTCommand', () => {
    it('shows help when no args or -h', () => {
      handleTensorRTCommand([]);
      expect(processExitSpy).toHaveBeenCalledWith(0);
      handleTensorRTCommand(['-h']);
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it('processes command', () => {
      handleTensorRTCommand(['model.onnx']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Exporting ONNX model to TensorRT Builder script: model.onnx...'),
      );
      handleTensorRTCommand(['']);
    });
  });
});
