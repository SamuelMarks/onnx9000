import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { handleOnnx2TfCommand } from '../src/commands/onnx2tf.js';

describe('handleOnnx2TfCommand', () => {
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
    await handleOnnx2TfCommand([]);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Usage: onnx9000 onnx2tf'));
    expect(processExitSpy).toHaveBeenCalledWith(0);

    await handleOnnx2TfCommand(['-h']);
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('Usage: onnx9000 onnx2tf'));
    expect(processExitSpy).toHaveBeenCalledWith(0);
  });

  it('should run onnx2tf command with default output', async () => {
    await handleOnnx2TfCommand(['model.onnx']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Loading ONNX model model.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Converting to TFLite format...');
    expect(consoleLogSpy).toHaveBeenCalledWith('Saving TFLite model to model.tflite...');
    expect(consoleLogSpy).toHaveBeenCalledWith('onnx2tf conversion complete.');
  });

  it('should run onnx2tf command with int8 and custom output', async () => {
    await handleOnnx2TfCommand(['model.onnx', '-o', 'out.tflite', '--int8']);
    expect(consoleLogSpy).toHaveBeenCalledWith('Loading ONNX model model.onnx...');
    expect(consoleLogSpy).toHaveBeenCalledWith(
      'Converting to TFLite format with INT8 quantization...',
    );
    expect(consoleLogSpy).toHaveBeenCalledWith('Saving TFLite model to out.tflite...');
  });
});
