import { expect, test, describe, vi } from 'vitest';
import {
  SafeTensors,
  SafetensorsError,
  SafetensorsOutOfBoundsError,
  SafetensorsOverlapError,
  SafetensorsInvalidJSONError,
  SafetensorsHeaderTooLargeError,
  SafetensorsShapeMismatchError,
  checkSafetensors,
  saveSafetensors,
  decodeFloat16,
  decodeBfloat16,
  getEndianness,
  swapEndianness,
  extractFromPyodideFS,
} from '../src/parser/safetensors.js';

import { validateOnnxShapesAndDtypes } from '../src/parser/safetensors.validator.js';
import { Graph } from '../src/ir/graph.js';
import { Tensor } from '../src/ir/tensor.js';

describe('Safetensors Typescript Parser', () => {
  test('save and load basic tensors', () => {
    const tensors = {
      a: new Uint8Array([1, 2, 3]),
      b: new Uint8Array([4, 5, 6]),
    };
    const buffer = saveSafetensors(tensors, { format: 'test' });
    expect(checkSafetensors(buffer.buffer)).toBe(true);

    const st = new SafeTensors(buffer.buffer);
    expect(st.metadata).toEqual({ format: 'test', version: '1.0' });
    expect(st.keys().sort()).toEqual(['a', 'b']);
    expect(st.getTensor('a')).toEqual(new Uint8Array([1, 2, 3]));
    expect(st.getTensor('b')).toEqual(new Uint8Array([4, 5, 6]));
  });

  test('getTypedArray', () => {
    const headerObj = {
      f32: {
        dtype: 'F32',
        shape: [2],
        data_offsets: [0, 8],
      },
      __metadata__: { format: 'pt' },
    };
    const headerStr = JSON.stringify(headerObj);
    let headerBytes = new TextEncoder().encode(headerStr);
    let pad = (8 - (headerBytes.byteLength % 8)) % 8;
    if (pad > 0) {
      headerBytes = new TextEncoder().encode(headerStr + ' '.repeat(pad));
    }

    const out = new Uint8Array(8 + headerBytes.byteLength + 8);
    const view = new DataView(out.buffer);
    view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
    out.set(headerBytes, 8);

    const floatData = new Float32Array([1.0, 2.0]);
    out.set(new Uint8Array(floatData.buffer), 8 + headerBytes.byteLength);

    const st = new SafeTensors(out.buffer);
    const floatArr = st.getTypedArray('f32');
    expect(floatArr instanceof Float32Array).toBe(true);
    expect(floatArr.length).toBe(2);
    expect(floatArr[0]).toBe(1.0);
    expect(floatArr[1]).toBe(2.0);
  });

  test('large header throws', () => {
    const out = new Uint8Array(10);
    const view = new DataView(out.buffer);
    view.setBigUint64(0, BigInt(101 * 1024 * 1024), true);
    expect(() => new SafeTensors(out.buffer)).toThrow(SafetensorsHeaderTooLargeError);
  });

  test('invalid json throws', () => {
    const out = new Uint8Array(8 + 4);
    const view = new DataView(out.buffer);
    view.setBigUint64(0, BigInt(4), true);
    out.set(new TextEncoder().encode('nope'), 8);
    expect(() => new SafeTensors(out.buffer)).toThrow(SafetensorsInvalidJSONError);
  });

  test('validate onnx shapes and dtypes', () => {
    const g = new Graph('test');
    g.tensors['w1'] = new Tensor(
      'w1',
      [2, 2],
      'float32',
      undefined,
      false,
      true,
      new Float32Array([1, 2, 3, 4]),
    );
    g.tensors['w2'] = new Tensor(
      'w2',
      [2],
      'int32',
      undefined,
      false,
      true,
      new Int32Array([1, 2]),
    );

    const headerObj = {
      w1: { dtype: 'F32', shape: [2], data_offsets: [0, 8] },
      w2: { dtype: 'F32', shape: [2], data_offsets: [8, 16] },
      __metadata__: {},
    };
    const headerStr = JSON.stringify(headerObj);
    let headerBytes = new TextEncoder().encode(headerStr);
    let pad = (8 - (headerBytes.byteLength % 8)) % 8;
    if (pad > 0) {
      headerBytes = new TextEncoder().encode(headerStr + ' '.repeat(pad));
    }

    const out = new Uint8Array(8 + headerBytes.byteLength + 16);
    const view = new DataView(out.buffer);
    view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
    out.set(headerBytes, 8);

    let logs: string[] = [];
    const originalWarn = console.warn;
    console.warn = (msg) => logs.push(msg);

    validateOnnxShapesAndDtypes(out.buffer, g);

    console.warn = originalWarn;

    expect(logs.some((m) => m.includes('Shape mismatch for w1'))).toBe(true);
    expect(logs.some((m) => m.includes('DType mismatch for w2'))).toBe(true);
  });

  test('1D Int8 padding issues', () => {
    const tensors = {
      a: new Uint8Array(new Int8Array([1, -1, 2, -2]).buffer),
    };
    const buffer = saveSafetensors(tensors);
    const st = new SafeTensors(buffer.buffer);
    // By default save is U8 for untyped bindings
    // But let's build custom header to test I8.
    const headerObj = {
      a: { dtype: 'I8', shape: [4], data_offsets: [0, 4] },
      __metadata__: {},
    };
    const headerStr = JSON.stringify(headerObj);
    let headerBytes = new TextEncoder().encode(headerStr);
    let pad = (8 - (headerBytes.byteLength % 8)) % 8;
    if (pad > 0) {
      headerBytes = new TextEncoder().encode(headerStr + ' '.repeat(pad));
    }

    const out = new Uint8Array(8 + headerBytes.byteLength + 4);
    const view = new DataView(out.buffer);
    view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
    out.set(headerBytes, 8);
    const data = new Int8Array([1, -1, 2, -2]);
    out.set(new Uint8Array(data.buffer), 8 + headerBytes.byteLength);

    const st2 = new SafeTensors(out.buffer);
    const arr = st2.getTypedArray('a');
    expect(arr instanceof Int8Array).toBe(true);
    expect(arr.length).toBe(4);
    expect(arr[0]).toBe(1);
    expect(arr[1]).toBe(-1);
    expect(arr[2]).toBe(2);
    expect(arr[3]).toBe(-2);
  });

  test('decode Float16', () => {
    // 0x3c00 is 1.0 in float16
    const f16 = new Uint16Array([0x3c00]);
    const f32 = decodeFloat16(f16);
    expect(f32[0]).toBeCloseTo(1.0);
  });

  test('decode Bfloat16', () => {
    // 0x3f80 is 1.0 in bfloat16
    const bf16 = new Uint16Array([0x3f80]);
    const f32 = decodeBfloat16(bf16);
    expect(f32[0]).toBeCloseTo(1.0);
  });
});
test('JS GC pointer protections', () => {
  const tensors = { a: new Uint8Array([1, 2]) };
  const buffer = saveSafetensors(tensors);
  const st = new SafeTensors(buffer.buffer);
  // By assigning explicit pointers without losing references
  const arr = st.getTypedArray('a');
  // This validates we can access memory cleanly without GC collecting the underlying array buffer
  // (V8 will not collect as long as TypedArray views hold a reference natively)
  expect(arr.length).toBe(2);
});

test('Benchmark 10,000 JSON keys parsing natively', async () => {
  const { benchmark10kKeys } = await import('../src/parser/safetensors.js');
  const res = await benchmark10kKeys();
  expect(res.keysParsed).toBe(10000);
  // It should parse 10k keys in less than 500ms
  expect(res.timeMs).toBeLessThan(500);
});
test('SharedArrayBuffer detection', () => {
  // Just verify checkSafetensors can accept SharedArrayBuffer without type error if running in supported environment
  if (typeof SharedArrayBuffer !== 'undefined') {
    const sab = new SharedArrayBuffer(100);
    expect(() => checkSafetensors(sab)).not.toThrow(TypeError);
  }
});

test('createBuffer gracefully falls back to ArrayBuffer', async () => {
  const { createBuffer } = await import('../src/parser/safetensors.js');
  const buf = createBuffer(10, true); // try to create SharedArrayBuffer if available
  expect(buf.byteLength).toBe(10);
  // It should either be an ArrayBuffer or a SharedArrayBuffer without throwing
  expect(
    buf instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer),
  ).toBe(true);
});

test('extractFromPyodideFS', () => {
  const FS = {
    lookupPath: (path: string) => {
      if (path === '/model.safetensors') {
        const tensors = { a: new Uint8Array([1, 2]) };
        const buffer = saveSafetensors(tensors);
        return { node: { contents: buffer } };
      }
      return { node: null };
    },
  };
  const st = extractFromPyodideFS(FS, '/model.safetensors');
  expect(st.keys()).toEqual(['a']);
});
test('XSS protection metadata', () => {
  const headerStr = JSON.stringify({
    a: { dtype: 'I8', shape: [1], data_offsets: [0, 1] },
    __metadata__: { malicious: '<script>alert(1)</script>' },
  });
  let headerBytes = new TextEncoder().encode(headerStr);
  let pad = (8 - (headerBytes.byteLength % 8)) % 8;
  if (pad > 0) headerBytes = new TextEncoder().encode(headerStr + ' '.repeat(pad));

  const out = new Uint8Array(8 + headerBytes.byteLength + 8);
  const view = new DataView(out.buffer);
  view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
  out.set(headerBytes, 8);
  expect(() => new SafeTensors(out.buffer)).toThrow(SafetensorsError);
});

test('DataView limits > 2GB', () => {
  // Just fake a very large offset and ensure it throws SafetensorsOutOfBoundsError
  // Needs to be 8 byte aligned, MAX_SAFE_INTEGER is 9007199254740991.
  // Let's use 9007199254740888
  const headerStr = JSON.stringify({
    a: { dtype: 'I8', shape: [200], data_offsets: [9007199254740888, 9007199254740888 + 200] },
    __metadata__: {},
  });
  const bytes = new TextEncoder().encode(headerStr);
  const out = new Uint8Array(8 + bytes.byteLength);
  const view = new DataView(out.buffer);
  view.setBigUint64(0, BigInt(bytes.byteLength), true);
  out.set(bytes, 8);
  expect(() => new SafeTensors(out.buffer)).toThrow(SafetensorsOutOfBoundsError);
});

test('getTensor bounds check manual', () => {
  const tensors = { a: new Uint8Array([1, 2]) };
  const buffer = saveSafetensors(tensors);
  const st = new SafeTensors(buffer.buffer);

  // Manually corrupt the tensor info to bypass constructor check
  // @ts-ignore
  st.tensors['a'].data_offsets = [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER + 10];

  expect(() => st.getTensor('a')).toThrow(SafetensorsOutOfBoundsError);
});

test('getTypedArray Big-Endian fallback', async () => {
  // Use vi.spyOn to mock getEndianness
  const mod = await import('../src/parser/safetensors.js');
  const spy = vi.spyOn(mod, 'getEndianness').mockReturnValue('BE');

  const headerObj = {
    a: { dtype: 'F32', shape: [1], data_offsets: [0, 4] },
    __metadata__: {},
  };
  const headerStr = JSON.stringify(headerObj);
  const headerBytes = new TextEncoder().encode(headerStr);

  const out = new Uint8Array(8 + headerBytes.byteLength + 4);
  const view = new DataView(out.buffer);
  view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
  out.set(headerBytes, 8);
  const floatData = new Float32Array([1.0]);
  out.set(new Uint8Array(floatData.buffer), 8 + headerBytes.byteLength);

  const st = new SafeTensors(out.buffer);

  // Should trigger Big-Endian swap logic
  const arr = st.getTypedArray('a');
  expect(arr instanceof Float32Array).toBe(true);

  spy.mockRestore();
});

test('getEndianness sanity check', () => {
  const endian = getEndianness();
  expect(endian === 'LE' || endian === 'BE').toBe(true);
});

test('swapEndianness', () => {
  const buf = new Uint8Array([0x12, 0x34, 0x56, 0x78]);
  swapEndianness(buf.buffer, buf.byteOffset, buf.byteLength, 4);
  expect(buf[0]).toBe(0x78);
  expect(buf[1]).toBe(0x56);
  expect(buf[2]).toBe(0x34);
  expect(buf[3]).toBe(0x12);
});
test('Data offsets overlap', () => {
  const headerStr = JSON.stringify({
    a: { dtype: 'I8', shape: [16], data_offsets: [0, 16] },
    b: { dtype: 'I8', shape: [16], data_offsets: [8, 24] },
    __metadata__: {},
  });
  let headerBytes = new TextEncoder().encode(headerStr);
  let pad = (8 - (headerBytes.byteLength % 8)) % 8;
  if (pad > 0) {
    headerBytes = new TextEncoder().encode(headerStr + ' '.repeat(pad));
  }

  const out = new Uint8Array(8 + headerBytes.byteLength + 24);
  const view = new DataView(out.buffer);
  view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
  out.set(headerBytes, 8);
  expect(() => new SafeTensors(out.buffer)).toThrow(SafetensorsOverlapError);
});
test('calculateVolume zero dim array', () => {
  const headerStr = JSON.stringify({
    a: { dtype: 'I8', shape: '1', data_offsets: [0, 1] },
    __metadata__: {},
  });
  let headerBytes = new TextEncoder().encode(headerStr);
  let pad = (8 - (headerBytes.byteLength % 8)) % 8;
  if (pad > 0) {
    headerBytes = new TextEncoder().encode(headerStr + ' '.repeat(pad));
  }

  const out = new Uint8Array(8 + headerBytes.byteLength + 8);
  const view = new DataView(out.buffer);
  view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
  out.set(headerBytes, 8);
  expect(() => new SafeTensors(out.buffer)).toThrow(SafetensorsShapeMismatchError);
});
test('1D Int8 zero byte tensors', () => {
  const tensors = {
    empty: new Uint8Array(0),
  };
  const buffer = saveSafetensors(tensors);
  const st = new SafeTensors(buffer.buffer);
  expect(st.getTensor('empty').length).toBe(0);
});

test('Number bounds check', () => {
  const headerObj = {
    a: {
      dtype: 'I8',
      shape: [2],
      data_offsets: [Number.MAX_SAFE_INTEGER + 1, Number.MAX_SAFE_INTEGER + 2],
    },
    __metadata__: {},
  };
  const headerStr = JSON.stringify(headerObj);
  let headerBytes = new TextEncoder().encode(headerStr);
  let pad = (8 - (headerBytes.byteLength % 8)) % 8;
  if (pad > 0) {
    headerBytes = new TextEncoder().encode(headerStr + ' '.repeat(pad));
  }

  const out = new Uint8Array(8 + headerBytes.byteLength);
  const view = new DataView(out.buffer);
  view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
  out.set(headerBytes, 8);

  expect(() => new SafeTensors(out.buffer)).toThrow(SafetensorsOutOfBoundsError);
});
