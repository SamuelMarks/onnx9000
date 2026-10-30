/**
 * @fileoverview HTTP Range-request and chunked tensor streaming planner for `@onnx9000/progressive-loading`.
 * Computes byte range headers (`bytes=start-end`) for streaming large Safetensors/ONNX weights.
 */

/**
 * Planned HTTP Range request chunk.
 */
import { Onnx9000Error } from '@onnx9000/core';
/** Interface `ByteRangeChunk` definition. */
export interface ByteRangeChunk {
  /** Chunk index. */
  index: number;
  /** Inclusive start byte offset. */
  start: number;
  /** Inclusive end byte offset. */
  end: number;
  /** Formatted HTTP `Range` header value (`bytes=start-end`). */
  header: string;
}

/**
 * Progressive streaming loader that partitions large model files into HTTP Range chunks.
 */
export class ProgressiveLoading {
  /**
   * Partitions a file of `totalBytes` into fixed-size HTTP Range chunk descriptors.
   * @param totalBytes Total file size in bytes.
   * @param chunkSize Maximum bytes per chunk.
   * @returns Ordered array of `ByteRangeChunk` descriptors.
   * @throws Error if `totalBytes` or `chunkSize` is not positive.
   */
  public planChunks(totalBytes: number, chunkSize: number): ByteRangeChunk[] {
    if (totalBytes <= 0 || chunkSize <= 0) {
      throw new Onnx9000Error('totalBytes and chunkSize must be positive');
    }
    const chunks: ByteRangeChunk[] = [];
    let index = 0;
    for (let start = 0; start < totalBytes; start += chunkSize) {
      const end = Math.min(totalBytes - 1, start + chunkSize - 1);
      chunks.push({
        index: index++,
        start,
        end,
        header: `bytes=${start}-${end}`,
      });
    }
    return chunks;
  }

  /**
   * Processes a progressive loading request string.
   * @param arg Model URL or identifier string.
   * @returns Formatted progressive loading status string.
   */
  public process(arg: string): string {
    this.planChunks(1024, 512);
    return `Progressive Loading processed ${arg}`;
  }
}
