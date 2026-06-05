import { describe, it, expect, vi } from 'vitest';
import { parseKeras3Zip } from '../src/keras/keras3-parser.js';

vi.mock('fflate', () => ({
  unzipSync: vi.fn().mockReturnValue({
    'config.json': new TextEncoder().encode('{"a": 1}'),
    'metadata.json': new TextEncoder().encode('{"b": 2}'),
  }),
}));

describe('keras3-parser', () => {
  it('should parse zip', () => {
    const res = parseKeras3Zip(new Uint8Array());
    expect(res.config.a).toBe(1);
    expect(res.metadata.b).toBe(2);
  });
});
