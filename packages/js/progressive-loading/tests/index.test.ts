import { describe, expect, it } from 'vitest';
import { ProgressiveLoading } from '../src/index.js';

describe('ProgressiveLoading', () => {
  it('should partition files into HTTP Range chunks and validate inputs', () => {
    const loader = new ProgressiveLoading();
    expect(loader.process('model.safetensors')).toBe(
      'Progressive Loading processed model.safetensors',
    );

    const chunks = loader.planChunks(1000, 400);
    expect(chunks).toEqual([
      { index: 0, start: 0, end: 399, header: 'bytes=0-399' },
      { index: 1, start: 400, end: 799, header: 'bytes=400-799' },
      { index: 2, start: 800, end: 999, header: 'bytes=800-999' },
    ]);

    expect(() => loader.planChunks(0, 100)).toThrow('totalBytes and chunkSize must be positive');
  });
});
