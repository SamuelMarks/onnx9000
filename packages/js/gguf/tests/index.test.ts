import { describe, expect, it } from 'vitest';
import { Gguf } from '../src/index.js';

describe('Gguf', () => {
  it('should roundtrip GGUF header and run self-check', () => {
    const gguf = new Gguf();
    expect(gguf.run()).toBe('[gguf] processed');

    const headerBytes = gguf.writeHeader();
    const parsed = gguf.readHeader(headerBytes);
    expect(parsed.magic).toBe('GGUF');
    expect(parsed.version).toBe(3);
    expect(parsed.tensorCount).toBe(0n);
    expect(parsed.kvCount).toBe(0n);
  });

  it('should reject truncated or invalid GGUF buffers', () => {
    const gguf = new Gguf();
    expect(() => gguf.readHeader(new Uint8Array(10))).toThrow(
      'GGUF buffer must be at least 24 bytes',
    );
    expect(() => gguf.readHeader(new Uint8Array(24))).toThrow('Invalid GGUF magic header');
  });
});
