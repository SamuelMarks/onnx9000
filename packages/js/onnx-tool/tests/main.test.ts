import { describe, it, expect } from 'vitest';
import { ONNXTool } from '../src/index';

describe('ONNXTool', () => {
  it('should process correctly', () => {
    const obj = new ONNXTool();
    expect(obj.process('test')).toBe('ONNX Tool processed test');
  });
});
