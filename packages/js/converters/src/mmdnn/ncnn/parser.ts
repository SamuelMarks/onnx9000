import { Tensor, Shape, DType } from '@onnx9000/core';

export interface NcnnNode {
  type: string;
  name: string;
  bottoms: string[];
  tops: string[];
  attrs: Record<string, string>;
}

export interface NcnnParam {
  magic: number;
  layerCount: number;
  blobCount: number;
  nodes: NcnnNode[];
}

export function parseNcnnParam(text: string): NcnnParam {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#'));

  let magic = 0;
  let layerCount = 0;
  let blobCount = 0;
  const nodes: NcnnNode[] = [];

  let lineIdx = 0;
  // Magic number usually starts with 7767517
  if (lines[lineIdx] === '7767517') {
    magic = 7767517;
    lineIdx++;
  }

  if (lineIdx < lines.length) {
    const currentLine = lines[lineIdx];
    if (!currentLine) return { magic, layerCount, blobCount, nodes };
    const counts = currentLine.split(/\s+/);
    layerCount = parseInt(counts[0] || '0', 10);
    blobCount = parseInt(counts[1] || '0', 10);
    lineIdx++;
  }

  for (; lineIdx < lines.length; lineIdx++) {
    const line = lines[lineIdx];
    if (!line) continue;
    const parts = line.split(/\s+/);
    if (parts.length < 4) continue;

    const type = parts[0] || '';
    const name = parts[1] || '';
    const bottomCount = parseInt(parts[2] || '0', 10);
    const topCount = parseInt(parts[3] || '0', 10);

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
      const pVal = parts[p];
      if (!pVal) continue;
      const kv = pVal.split('=');
      if (kv.length === 2) {
        if (kv[0]) attrs[kv[0]] = kv[1] || '';
      } else if (kv.length === 1 && pVal.startsWith('-')) {
        // arrays usually like -23309=val
        const arrayKey = pVal.split('=');
        if (arrayKey.length === 2) {
          if (arrayKey[0]) attrs[arrayKey[0]] = arrayKey[1] || '';
        }
      }
    }

    nodes.push({ type, name, bottoms, tops, attrs });
  }

  return { magic, layerCount, blobCount, nodes };
}

// Simple binary parser that just advances an offset and returns Tensors
export class NcnnBinParser {
  private view: DataView;
  private offset = 0;

  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
  }

  readFloats(count: number): Float32Array {
    const floats = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      floats[i] = this.view.getFloat32(this.offset, true); // NCNN is little-endian
      this.offset += 4;
    }
    return floats;
  }

  readInts(count: number): Int32Array {
    const ints = new Int32Array(count);
    for (let i = 0; i < count; i++) {
      ints[i] = this.view.getInt32(this.offset, true);
      this.offset += 4;
    }
    return ints;
  }

  readBytes(count: number): Uint8Array {
    const bytes = new Uint8Array(this.view.buffer, this.view.byteOffset + this.offset, count);
    this.offset += count;
    // align to 4 bytes for ncnn
    const rem = this.offset % 4;
    if (rem !== 0) {
      this.offset += 4 - rem;
    }
    return bytes;
  }
}
