import { describe, it, expect, vi } from 'vitest';
import { convert } from '../../src/mmdnn/api.js';

vi.mock('../../src/mmdnn/file-loader.js', () => ({
  FileLoader: class {
    initialize() {}
  },
}));

describe('mmdnn api', () => {
  it('should convert onnx to pytorch', async () => {
    const f = new File([''], 'test.onnx');
    f.arrayBuffer = async () => new Uint8Array([8, 0]).buffer;
    const res = await convert('onnx', 'pytorch_code', [f]);
    expect(res).toContain('class ONNXModel(nn.Module):');
  });
});
