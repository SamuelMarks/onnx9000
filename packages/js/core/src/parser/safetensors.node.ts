import { SafetensorsHeaderTooLargeError, SafetensorsInvalidJSONError } from './safetensors.js';

// We dynamically import fs so browser environments don't crash
export function readSafetensorsHeaderSync(fd: number, fs: any) {
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
  let headerObj: Record<string, any>;
  try {
    headerObj = JSON.parse(headerStr);
  } catch (e) {
    throw new SafetensorsInvalidJSONError(`Invalid JSON header: ${e}`);
  }

  return { headerObj, headerSize };
}

export function readSafetensorsChunkSync(
  fd: number,
  fs: any,
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

export function saveSafetensorsFileSync(
  filename: string,
  fs: any,
  tensors: Record<string, Uint8Array | { data: Uint8Array; dtype?: string; shape?: number[] }>,
  metadata?: Record<string, string>,
) {
  try {
    const bytes = saveSafetensors(tensors, metadata);
    fs.writeFileSync(filename, Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  } catch (e: any) {
    if (e.code === 'ENOSPC' || e.message.includes('ENOSPC')) {
      throw new Error(`SafetensorsWriteError: disk space exhausted writing ${filename}`);
    }
    throw e;
  }
}
