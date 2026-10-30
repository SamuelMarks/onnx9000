import * as fflate from 'fflate';
import { describe, expect, it, vi } from 'vitest';
import { parseKeras3Zip } from '../../src/keras/keras3-parser.js';

vi.mock('fflate', () => ({
  unzipSync: vi.fn(),
}));

describe('keras3-parser.ts', () => {
  it('should call and cover parseKeras3Zip across all file types and branches', () => {
    vi.mocked(fflate.unzipSync).mockReturnValueOnce({
      'sub/config.json': new TextEncoder().encode(JSON.stringify({ class_name: 'Sequential' })),
      'sub/metadata.json': new TextEncoder().encode(JSON.stringify({ keras_version: '3.0' })),
      'sub/model.weights.h5': new Uint8Array([1, 2]),
      'sub/model.weights.safetensors': new Uint8Array([3, 4]),
      'other.txt': new Uint8Array([5]),
    });
    const res1 = parseKeras3Zip(new Uint8Array([1]));
    expect(res1.config.class_name).toBe('Sequential');
    expect(res1.weightsH5).toBeDefined();
    expect(res1.weightsSafetensors).toBeDefined();

    vi.mocked(fflate.unzipSync).mockReturnValueOnce({
      'config.json': new TextEncoder().encode(JSON.stringify({ class_name: 'Functional' })),
      'metadata.json': new TextEncoder().encode(JSON.stringify({ keras_version: '3.1' })),
      'model.weights.h5': new Uint8Array([1]),
      'model.weights.safetensors': new Uint8Array([2]),
    });
    const res2 = parseKeras3Zip(new Uint8Array([1]));
    expect(res2.config.class_name).toBe('Functional');

    vi.mocked(fflate.unzipSync).mockReturnValueOnce({
      'config.json': new TextEncoder().encode(JSON.stringify({ class_name: 'Minimal' })),
    });
    const res3 = parseKeras3Zip(new Uint8Array([1]));
    expect(res3.metadata).toEqual({});

    vi.mocked(fflate.unzipSync).mockReturnValueOnce({
      'other.txt': new Uint8Array([1]),
    });
    expect(() => parseKeras3Zip(new Uint8Array([1]))).toThrow(/missing config.json/);
  });
});
