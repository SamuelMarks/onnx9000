import { describe, expect, it } from 'vitest';
import { ONNXTool } from '../src/index.js';

describe('ONNXTool', () => {
  it('should compute MACs, FLOPs, and parameter counts', () => {
    const tool = new ONNXTool();
    expect(tool.process('resnet50')).toBe('ONNX Tool processed resnet50');

    const report = tool.profileLayers([
      { opType: 'Gemm', weightShape: [4, 8], outputShape: [1, 4] },
      { opType: 'Relu', weightShape: [], outputShape: [] },
    ]);
    expect(report.totalParams).toBe(32);
    expect(report.totalMacs).toBe(128);
    expect(report.totalFlops).toBe(256);
  });
});
