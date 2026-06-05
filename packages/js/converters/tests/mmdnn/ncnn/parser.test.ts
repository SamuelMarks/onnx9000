import { describe, it, expect } from 'vitest';
import { parseNcnnParam, NcnnBinParser } from '../../../src/mmdnn/ncnn/parser.js';

describe('ncnn parser', () => {
  it('should parse param', () => {
    const res = parseNcnnParam('7767517\n1 1\nInput in 0 1 out 0=1');
    expect(res.magic).toBe(7767517);
    expect(res.layerCount).toBe(1);
    expect(res.nodes[0].type).toBe('Input');
  });

  it('should read bin', () => {
    const bin = new NcnnBinParser(new Uint8Array([0, 0, 0, 0]).buffer);
    const floats = bin.readFloats(1);
    expect(floats[0]).toBe(0);
  });
});
