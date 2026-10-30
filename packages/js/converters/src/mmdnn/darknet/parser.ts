/**
 * @fileoverview Darknet `.cfg` configuration parser and binary `.weights` loader.
 */
import type { DarknetSection } from '../types.js';

/** Alias for `DarknetSection` representing a parsed Darknet configuration block. */
export type DarknetLayer = DarknetSection;
export type { DarknetSection };

/**
 * Parses a Darknet `.cfg` string into an ordered array of `DarknetSection` descriptors.
 * @param cfgStr Raw Darknet `.cfg` text content.
 * @returns Array of parsed `DarknetSection` objects.
 */
export function parseCfg(cfgStr: string): DarknetSection[] {
  const lines = cfgStr.split(String.fromCharCode(10));
  const layers: DarknetSection[] = [];
  let currentLayer: DarknetSection | null = null;

  for (const rawLine of lines) {
    const line = (rawLine.split('#')[0] || '').trim();
    if (!line) continue;

    if (line.startsWith('[') && line.endsWith(']')) {
      const type = line.substring(1, line.length - 1).trim();
      currentLayer = { type };
      layers.push(currentLayer);
    } else if (currentLayer) {
      const parts = line.split('=');
      if (parts.length >= 2) {
        const key = parts[0]?.trim();
        const valueStr = parts.slice(1).join('=').trim();
        if (!key) continue;

        if (valueStr.includes(',')) {
          const listVals = valueStr.split(',').map((v) => {
            const vTrim = v.trim();
            const n = Number(vTrim);
            return Number.isNaN(n) ? vTrim : n;
          });
          currentLayer[key] = listVals;
        } else {
          const n = Number(valueStr);
          if (!Number.isNaN(n) && valueStr !== '') {
            currentLayer[key] = n;
          } else {
            currentLayer[key] = valueStr;
          }
        }
      }
    }
  }

  return layers;
}

/**
 * Parses a Darknet binary `.weights` buffer into a `Float32Array` of weight values.
 * @param buffer Raw `ArrayBuffer` containing Darknet binary header and float weights.
 * @returns `Float32Array` view over the weight payload.
 */
export function parseWeights(buffer: ArrayBuffer): Float32Array {
  const view = new DataView(buffer);
  if (buffer.byteLength < 16) {
    return new Float32Array(0);
  }

  const major = view.getInt32(0, true);
  const minor = view.getInt32(4, true);

  let offset = 16;
  if (major * 10 + minor >= 2 && major < 1000 && minor < 1000) {
    offset = 20;
  }

  if (offset > buffer.byteLength) offset = buffer.byteLength;

  return new Float32Array(buffer, offset);
}
