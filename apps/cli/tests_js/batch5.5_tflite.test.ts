import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { handleTfliteCommand } from '../src/commands/tflite.js';
import * as onnx2tf from '../src/commands/onnx2tf.js';

vi.mock('../src/commands/onnx2tf.js', () => ({
  handleOnnx2TfCommand: vi.fn(),
}));

describe('CLI Commands Batch 5', () => {
  beforeEach(() => {
    vi.mocked(onnx2tf.handleOnnx2TfCommand).mockClear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('handleTfliteCommand', () => {
    it('aliases to onnx2tf command', async () => {
      await handleTfliteCommand(['model.onnx']);
      expect(onnx2tf.handleOnnx2TfCommand).toHaveBeenCalledWith(['model.onnx']);
    });
  });
});
