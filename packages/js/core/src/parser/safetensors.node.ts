import {
  SafetensorsHeaderTooLargeError,
  SafetensorsInvalidJSONError,
  TensorInfo,
} from './safetensors.js';

/**
 * Minimal interface for Node.js 'fs' module.
 */
export interface NodeFS {
  readSync(
    fd: number,
    buffer: Uint8Array,
    offset: number,
    length: number,
    position: number | null,
  ): number;
  writeFileSync(
    path: string | number,
    data: Uint8Array,
    options?: { encoding?: string | null; mode?: number | string; flag?: string } | string | null,
  ): void;
}

/**
 * Synchronously reads the header of a safetensors file.
 * @param fd File descriptor
 * @param fs Node.js fs module
 * @returns Parsed header and its size
 */
export function readSafetensorsHeaderSync(fd: number, fs: NodeFS) {
  // Read first 8 bytes
  const buf8 = Buffer.alloc(8);
  fs.readSync(fd, buf8, 0, 8, 0);

  const headerSizeBig = buf8.readBigUInt64LE(0);
  if (headerSizeBig > BigInt(100 * 1024 * 1024)) {
    throw new SafetensorsHeaderTooLargeError('Header size exceeds 100MB');
  }
  const headerSize = Number(headerSizeBig);

  // Read header
  const bufHeader = Buffer.alloc(headerSize);
  fs.readSync(fd, bufHeader, 0, headerSize, 8);

  const headerStr = bufHeader.toString('utf-8');
  let headerObj: Record<string, TensorInfo | Record<string, string>>;
  try {
    headerObj = JSON.parse(headerStr);
  } catch (e) {
    throw new SafetensorsInvalidJSONError(`Invalid JSON header: ${e}`);
  }

  return { headerObj, headerSize };
}

/**
 * Synchronously reads a chunk of data from a safetensors file.
 * @param fd File descriptor
 * @param fs Node.js fs module
 * @param headerSize Size of the header
 * @param begin Offset start
 * @param end Offset end
 * @returns Chunk data as Uint8Array
 */
export function readSafetensorsChunkSync(
  fd: number,
  fs: NodeFS,
  headerSize: number,
  begin: number,
  end: number,
): Uint8Array {
  const absBegin = 8 + headerSize + begin;
  const length = end - begin;
  const buf = Buffer.alloc(length);
  fs.readSync(fd, buf, 0, length, absBegin);
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

import { saveSafetensors } from './safetensors.js';

/**
 * Synchronously saves tensors into a safetensors file.
 * @param filename Target filename
 * @param fs Node.js fs module
 * @param tensors Tensors to save
 * @param metadata Optional metadata
 */
export function saveSafetensorsFileSync(
  filename: string,
  fs: NodeFS,
  tensors: Record<string, Uint8Array | { data: Uint8Array; dtype?: string; shape?: number[] }>,
  metadata?: Record<string, string>,
) {
  try {
    const bytes = saveSafetensors(tensors, metadata);
    fs.writeFileSync(filename, Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  } catch (e) {
    const error = e as Error & { code?: string };
    if (error.code === 'ENOSPC' || error.message.includes('ENOSPC')) {
      throw new Error(`SafetensorsWriteError: disk space exhausted writing ${filename}`);
    }
    throw error;
  }
}
