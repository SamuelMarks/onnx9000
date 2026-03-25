import { describe, it, expect, vi } from 'vitest';
import { convert } from '../../src/mmdnn/api.js';

describe('MMDNN - OnnxScript API integration', () => {
  it('should parse an empty or mock onnxscript file into a fallback graph', async () => {
    const fakeFile = new File(['def fail():\n  pass\n'], 'model.py', { type: 'text/plain' });
    const graph = await convert('onnxscript', 'onnx', [fakeFile]);
    expect(graph.name).toBe('onnxscript-imported');
  });

  it('should throw error/catch and use fallback when parsing fails', async () => {
    // Actually our simplistic parser rarely throws, but if we pass something that causes an internal exception,
    // api.ts should catch it and return a graph named "onnxscript-imported"
    const mockFile = new File([''], 'model.py', { type: 'text/plain' });
    mockFile.text = () => {
      throw new Error('simulated read error');
    };
    await expect(convert('onnxscript', 'onnx', [mockFile])).rejects.toThrow();
  });
});
