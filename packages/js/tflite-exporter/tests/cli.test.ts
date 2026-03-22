import { describe, it, expect, vi } from 'vitest';
import { onnx2tfCli } from '../src/cli/index';
import fs from 'fs';

vi.mock('fs', () => ({
  default: {
    readFileSync: vi.fn().mockReturnValue(new Uint8Array([0, 1, 2])),
    writeFileSync: vi.fn(),
  },
}));

describe('TFLite Compiler - CLI', () => {
  it('should error without inputs', async () => {
    const mockExit = vi.spyOn(process, 'exit').mockImplementation(() => undefined as never);
    const mockError = vi.spyOn(console, 'error').mockImplementation(() => {});

    await onnx2tfCli([]);

    expect(mockError).toHaveBeenCalledWith('Error: Must provide an input .onnx file.');
    expect(mockExit).toHaveBeenCalledWith(1);

    mockExit.mockRestore();
    mockError.mockRestore();
  });

  it('should process normal file with output and flags', async () => {
    const mockLog = vi.spyOn(console, 'log').mockImplementation(() => {});

    await onnx2tfCli(['model.onnx', '-o', 'out.tflite', '--keep-nchw', '--fp16']);

    expect(mockLog).toHaveBeenCalledWith(
      expect.stringContaining('Loading ONNX model from model.onnx'),
    );
    expect(mockLog).toHaveBeenCalledWith(
      expect.stringContaining('Compiling to TFLite... (keepNchw=true, quantMode=fp16)'),
    );

    mockLog.mockRestore();
  });

  it('should process disable optimization flag', async () => {
    const mockLog = vi.spyOn(console, 'log').mockImplementation(() => {});

    await onnx2tfCli(['model.onnx', '--disable-optimization']);

    expect(mockLog).toHaveBeenCalledWith(
      expect.stringContaining('Disabling layout and math optimizations...'),
    );

    mockLog.mockRestore();
  });

  it('should process int8 mode', async () => {
    const mockLog = vi.spyOn(console, 'log').mockImplementation(() => {});

    await onnx2tfCli(['model.onnx', '--int8', '-b', '4']);

    expect(mockLog).toHaveBeenCalledWith(expect.stringContaining('quantMode=int8'));

    mockLog.mockRestore();
  });

  it('should process saved-model mode', async () => {
    const mockLog = vi.spyOn(console, 'log').mockImplementation(() => {});

    await onnx2tfCli(['model.onnx', '--saved-model']);

    expect(mockLog).toHaveBeenCalledWith(
      expect.stringContaining('Generating TensorFlow SavedModel Protobuf...'),
    );

    mockLog.mockRestore();
  });
});
