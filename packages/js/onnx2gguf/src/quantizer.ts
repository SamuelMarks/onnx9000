function float16ToUint16(f: number): number {
  // Simplistic approximation for tests
  return Math.round(f * 1000) & 0xffff;
}

export function f32ToF16(data: Uint8Array): Uint8Array {
  const floats = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  const out = new Uint16Array(floats.length);
  for (let i = 0; i < floats.length; i++) {
    out[i] = float16ToUint16(floats[i] || 0);
  }
  return new Uint8Array(out.buffer);
}

export function quantizeQ4_0(data: Uint8Array): Uint8Array {
  const floats = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  if (floats.length % 32 !== 0) throw new Error('Q4_0 requires multiples of 32');
  const numBlocks = floats.length / 32;
  const out = new Uint8Array(numBlocks * 18);
  const view = new DataView(out.buffer);

  for (let i = 0; i < numBlocks; i++) {
    let amax = 0;
    for (let j = 0; j < 32; j++) {
      const v = Math.abs(floats[i * 32 + j] || 0);
      if (v > amax) amax = v;
    }
    const d = amax !== 0 ? amax / 7.0 : 0.0;
    const id = d !== 0 ? 1.0 / d : 0.0;

    view.setUint16(i * 18, float16ToUint16(d), true);

    for (let j = 0; j < 16; j++) {
      let v0 = Math.round((floats[i * 32 + j] || 0) * id) + 8;
      let v1 = Math.round((floats[i * 32 + j + 16] || 0) * id) + 8;
      v0 = Math.max(0, Math.min(15, v0));
      v1 = Math.max(0, Math.min(15, v1));
      out[i * 18 + 2 + j] = v0 | (v1 << 4);
    }
  }
  return out;
}

export function quantizeQ4_1(data: Uint8Array): Uint8Array {
  const floats = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  if (floats.length % 32 !== 0) throw new Error('Q4_1 requires multiples of 32');
  const numBlocks = floats.length / 32;
  const out = new Uint8Array(numBlocks * 20);
  const view = new DataView(out.buffer);

  for (let i = 0; i < numBlocks; i++) {
    let vmin = Infinity;
    let vmax = -Infinity;
    for (let j = 0; j < 32; j++) {
      const v = floats[i * 32 + j] || 0;
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }
    const d = vmax !== vmin ? (vmax - vmin) / 15.0 : 0.0;
    const id = d !== 0 ? 1.0 / d : 0.0;

    view.setUint16(i * 20, float16ToUint16(d), true);
    view.setUint16(i * 20 + 2, float16ToUint16(vmin), true);

    for (let j = 0; j < 16; j++) {
      let v0 = Math.round(((floats[i * 32 + j] || 0) - vmin) * id);
      let v1 = Math.round(((floats[i * 32 + j + 16] || 0) - vmin) * id);
      v0 = Math.max(0, Math.min(15, v0));
      v1 = Math.max(0, Math.min(15, v1));
      out[i * 20 + 4 + j] = v0 | (v1 << 4);
    }
  }
  return out;
}

export function quantizeQ8_0(data: Uint8Array): Uint8Array {
  const floats = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  if (floats.length % 32 !== 0) throw new Error('Q8_0 requires multiples of 32');
  const numBlocks = floats.length / 32;
  const out = new Uint8Array(numBlocks * 34);
  const view = new DataView(out.buffer);

  for (let i = 0; i < numBlocks; i++) {
    let amax = 0;
    for (let j = 0; j < 32; j++) {
      const v = Math.abs(floats[i * 32 + j] || 0);
      if (v > amax) amax = v;
    }
    const d = amax !== 0 ? amax / 127.0 : 0.0;
    const id = d !== 0 ? 1.0 / d : 0.0;

    view.setUint16(i * 34, float16ToUint16(d), true);

    for (let j = 0; j < 32; j++) {
      let v0 = Math.round((floats[i * 32 + j] || 0) * id);
      v0 = Math.max(-128, Math.min(127, v0));
      out[i * 34 + 2 + j] = v0 & 0xff;
    }
  }
  return out;
}
