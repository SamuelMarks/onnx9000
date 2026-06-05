import { describe, it, expect } from 'vitest';
import { parseCfg } from '../../../src/mmdnn/darknet/parser.js';

describe('darknet/parser', () => {
  it('should parse cfg', () => {
    const cfg = `
    [net]
    channels=3
    [convolutional]
    filters=16
    `;
    const res = parseCfg(cfg);
    expect(res.length).toBe(2);
    expect(res[0].type).toBe('net');
    expect(res[1].type).toBe('convolutional');
    expect((res[1] as any).filters).toBe(16);
  });
});
