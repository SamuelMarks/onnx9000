import { describe, it, expect } from 'vitest';
import { parsePrototxt } from '../src/mmdnn/caffe/parser.js';

describe('caffe/parser', () => {
  it('should parse prototxt', () => {
    const text = `
    name: "Test"
    layer {
      name: "relu1"
      type: "ReLU"
    }
    `;
    const res: any = parsePrototxt(text);
    expect(res.layer).toBeDefined();
    expect(res.layer.length).toBe(1);
    expect(res.layer[0].type).toBe('ReLU');
  });
});
