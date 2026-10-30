/**
 * @fileoverview NCNN `.param` text parser and `.bin` binary reader.
 */
import type { NcnnLayer } from '../types.js';

/** Alias for `NcnnLayer` representing a parsed NCNN node. */
export type NcnnNode = NcnnLayer;
export type { NcnnLayer };

/** Parsed NCNN `.param` structure. */
export interface NcnnParam {
  /** Magic header number (typically `7767517`). */
  magic: number;
  /** Declared layer count. */
  layerCount: number;
  /** Declared blob count. */
  blobCount: number;
  /** Parsed NCNN layer nodes. */
  nodes: NcnnLayer[];
}

/**
 * Parses an NCNN `.param` text string into an `NcnnParam` structure.
 * @param text Raw text from an NCNN `.param` file.
 * @returns Parsed `NcnnParam` object.
 */
export function parseNcnnParam(text: string): NcnnParam {
  const lines = text
    .split(String.fromCharCode(10))
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#'));

  let magic = 0;
  let layerCount = 0;
  let blobCount = 0;
  const nodes: NcnnLayer[] = [];

  let lineIdx = 0;
  if (lines[lineIdx] === '7767517') {
    magic = 7767517;
    lineIdx++;
  }

  const currentLine = lines[lineIdx] || '0';
  const counts = currentLine.split(/\s+/);
  layerCount = parseInt(counts[0] as string, 10);
  blobCount = parseInt(counts[1] || '0', 10);
  lineIdx++;

  for (const line of lines.slice(lineIdx)) {
    const parts = line.split(/\s+/);
    if (parts.length < 4) continue;

    const type = parts[0] as string;
    const name = parts[1] as string;
    const bottomCount = parseInt(parts[2] as string, 10);
    const topCount = parseInt(parts[3] as string, 10);

    let p = 4;
    const bottoms: string[] = [];
    for (let i = 0; i < bottomCount; i++) {
      bottoms.push(parts[p++] || '');
    }

    const tops: string[] = [];
    for (let i = 0; i < topCount; i++) {
      tops.push(parts[p++] || '');
    }

    const attrs: Record<string, string> = {};
    for (; p < parts.length; p++) {
      const pVal = parts[p] as string;
      const kv = pVal.split('=');
      if (kv.length === 2 && kv[0]) {
        attrs[kv[0]] = kv[1] || '';
      }
    }

    nodes.push({ type, name, bottoms, tops, attrs });
  }

  return { magic, layerCount, blobCount, nodes };
}

/**
 * Sequential binary reader for NCNN `.bin` weight buffers.
 */
export class NcnnBinParser {
  /** Property `view` of `NcnnBinParser`. */
  private view: DataView;
  /** Property `offset` of `NcnnBinParser`. */
  private offset = 0;

  /**
   * Initializes a new `NcnnBinParser` over an `ArrayBuffer`.
   * @param buffer Binary buffer containing NCNN weights.
   */
  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
  }

  /**
   * Reads `count` little-endian 32-bit floats.
   * @param count Number of float32 elements to read.
   * @returns `Float32Array` of read values.
   */
  readFloats(count: number): Float32Array {
    const floats = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      floats[i] = this.view.getFloat32(this.offset, true);
      this.offset += 4;
    }
    return floats;
  }

  /**
   * Reads `count` little-endian 32-bit signed integers.
   * @param count Number of int32 elements to read.
   * @returns `Int32Array` of read values.
   */
  readInts(count: number): Int32Array {
    const ints = new Int32Array(count);
    for (let i = 0; i < count; i++) {
      ints[i] = this.view.getInt32(this.offset, true);
      this.offset += 4;
    }
    return ints;
  }

  /**
   * Reads `count` bytes and aligns the cursor to a 4-byte boundary.
   * @param count Number of bytes to read.
   * @returns `Uint8Array` view over the read bytes.
   */
  readBytes(count: number): Uint8Array {
    const bytes = new Uint8Array(this.view.buffer, this.view.byteOffset + this.offset, count);
    this.offset += count;
    const rem = this.offset % 4;
    if (rem !== 0) {
      this.offset += 4 - rem;
    }
    return bytes;
  }
}
