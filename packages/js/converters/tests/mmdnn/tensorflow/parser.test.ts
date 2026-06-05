import { describe, it, expect } from 'vitest';
import { parsePbtxt, parseTFProto } from '../../../src/mmdnn/tensorflow/parser.js';

describe('tf/parser', () => {
  it('should parse pbtxt', () => {
    const pbtxt = `
    node {
      name: "input"
      op: "Placeholder"
    }
    `;
    const res = parsePbtxt(pbtxt);
    expect(res.node.length).toBe(1);
    expect(res.node[0].op).toBe('Placeholder');
    expect(parseTFProto(new Uint8Array())).toBeDefined();
  });
});
