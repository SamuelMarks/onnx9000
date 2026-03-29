/* eslint-disable */
// @ts-nocheck
export interface DarknetLayer {
  type: string;
  [key: string]: object;
}

export function parseCfg(cfgStr: string): DarknetLayer[] {
  const lines = cfgStr.split('\n');
  const layers: DarknetLayer[] = [];
  let currentLayer: DarknetLayer | null = null;

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    if (rawLine === undefined) continue;

    // Remove comments
    const line = rawLine.split('#')[0]!.trim();
    if (!line) continue;

    if (line.startsWith('[') && line.endsWith(']')) {
      const type = line.substring(1, line.length - 1).trim();
      currentLayer = { type };
      layers.push(currentLayer);
    } else if (currentLayer) {
      const parts = line.split('=');
      if (parts.length >= 2) {
        const key = parts[0]!.trim();
        const valueStr = parts.slice(1).join('=').trim();

        if (valueStr.includes(',')) {
          const listVals = valueStr.split(',').map((v) => {
            const vTrim = v.trim();
            const n = Number(vTrim);
            return isNaN(n) ? vTrim : n;
          });
          currentLayer[key] = listVals;
        } else {
          const n = Number(valueStr);
          if (!isNaN(n) && valueStr !== '') {
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

export function parseWeights(buffer: ArrayBuffer): Float32Array {
  const view = new DataView(buffer);
  if (buffer.byteLength < 16) {
    return new Float32Array(0);
  }

  const major = view.getInt32(0, true);
  const minor = view.getInt32(4, true);

  // If major*10 + minor >= 2 and it's a realistic version (not random data)
  // then seen is int64, so offset = 20. Otherwise 16.
  let offset = 16;
  if (major * 10 + minor >= 2 && major < 1000 && minor < 1000) {
    offset = 20;
  }

  if (offset > buffer.byteLength) offset = buffer.byteLength;

  return new Float32Array(buffer, offset);
}
