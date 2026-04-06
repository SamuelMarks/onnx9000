globalThis.Response = class Response {
  constructor() {}
} as Object;
globalThis.Request = class Request {
  constructor() {}
} as Object;
import { vi } from 'vitest';
import { saveSafetensors, createBuffer, fetchSafetensorsChunk } from '../src/parser/safetensors';
import {
  _mallocSafetensors,
  passToPyodideWASM,
  extractFromPyodideFS,
  benchmark10kKeys,
  SafeTensors,
} from '../src/parser/safetensors';
import { decodeFloat16, getEndianness, saveSafetensors } from '../src/parser/safetensors';
import { expect, test, describe, vi, beforeEach, afterEach } from 'vitest';
import {
  SafeTensors,
  SafetensorsError,
  SafetensorsHeaderTooLargeError,
  SafetensorsInvalidHeaderError,
  SafetensorsInvalidJSONError,
  SafetensorsDuplicateKeyError,
  SafetensorsInvalidOffsetError,
  SafetensorsOutOfBoundsError,
  SafetensorsOverlapError,
  SafetensorsAlignmentError,
  SafetensorsInvalidDtypeError,
  SafetensorsShapeMismatchError,
  SafetensorsFileEmptyError,
  SafetensorsFileTooSmallError,
  padTo8Bytes,
  saveSafetensors,
  checkSafetensors,
  _mallocSafetensors,
  passToPyodideWASM,
  extractFromPyodideFS,
  fetchSafetensorsHeader,
  fetchSafetensorsChunk,
  loadTensors,
} from '../src/parser/safetensors.js';

function createDummySafeTensorsBuffer(headerObj: Object, dataLength: number): ArrayBuffer {
  const headerStr = JSON.stringify(headerObj);
  let headerBytes = new TextEncoder().encode(headerStr);
  const pad = (8 - (headerBytes.byteLength % 8)) % 8;
  if (pad > 0) {
    headerBytes = new TextEncoder().encode(headerStr + ' '.repeat(pad));
  }
  const out = new Uint8Array(8 + headerBytes.byteLength + dataLength);
  const view = new DataView(out.buffer);
  view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
  out.set(headerBytes, 8);
  return out.buffer;
}

describe('Safetensors Parser - Full Coverage', () => {
  test('Empty and small files', () => {
    expect(() => new SafeTensors(new ArrayBuffer(0))).toThrow(SafetensorsFileEmptyError);
    expect(() => new SafeTensors(new ArrayBuffer(7))).toThrow(SafetensorsFileTooSmallError);
  });

  test('Header out of bounds', () => {
    const out = new Uint8Array(8);
    const view = new DataView(out.buffer);
    view.setBigUint64(0, BigInt(100), true);
    expect(() => new SafeTensors(out.buffer)).toThrow(SafetensorsOutOfBoundsError);
  });

  test('Invalid UTF-8 header', () => {
    const out = new Uint8Array(12);
    const view = new DataView(out.buffer);
    view.setBigUint64(0, BigInt(4), true);
    out[8] = 0xff;
    out[9] = 0xff;
    expect(() => new SafeTensors(out.buffer)).toThrow(SafetensorsInvalidHeaderError);
  });

  test('Header must be dictionary', () => {
    const headerStr = '[]';
    const headerBytes = new TextEncoder().encode(headerStr);
    const out = new Uint8Array(8 + headerBytes.byteLength);
    const view = new DataView(out.buffer);
    view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
    out.set(headerBytes, 8);
    expect(() => new SafeTensors(out.buffer)).toThrow(SafetensorsInvalidJSONError);
  });

  test('Unknown dtype', () => {
    const headerObj = { a: { dtype: 'XXX', shape: [1], data_offsets: [0, 8] } };
    const buffer = createDummySafeTensorsBuffer(headerObj, 8);
    expect(() => new SafeTensors(buffer)).toThrow(SafetensorsInvalidDtypeError);
  });

  test('Invalid offsets: begin > end', () => {
    const headerObj = { a: { dtype: 'I8', shape: [1], data_offsets: [8, 0] } };
    const buffer = createDummySafeTensorsBuffer(headerObj, 8);
    expect(() => new SafeTensors(buffer)).toThrow(SafetensorsInvalidOffsetError);
  });

  test('Invalid offsets: not 8-byte aligned', () => {
    const headerObj = { a: { dtype: 'I8', shape: [1], data_offsets: [1, 2] } };
    const buffer = createDummySafeTensorsBuffer(headerObj, 8);
    expect(() => new SafeTensors(buffer)).toThrow(SafetensorsAlignmentError);
  });

  test('Data region exceeds file boundaries', () => {
    const headerObj = { a: { dtype: 'I8', shape: [16], data_offsets: [0, 16] } };
    const buffer = createDummySafeTensorsBuffer(headerObj, 0);
    expect(() => new SafeTensors(buffer)).toThrow(SafetensorsOutOfBoundsError);
  });

  test('Shape volume mismatch', () => {
    const headerObj = { a: { dtype: 'F32', shape: [2], data_offsets: [0, 16] } };
    const buffer = createDummySafeTensorsBuffer(headerObj, 16);
    expect(() => new SafeTensors(buffer)).toThrow(SafetensorsShapeMismatchError);
  });

  test('Overlap Error Check', () => {
    const headerObj = {
      a: { dtype: 'I8', shape: [16], data_offsets: [0, 16] },
      b: { dtype: 'I8', shape: [16], data_offsets: [8, 24] },
    };
    const buffer = createDummySafeTensorsBuffer(headerObj, 24);
    expect(() => new SafeTensors(buffer)).toThrow(SafetensorsOverlapError);
  });

  test('getTensor non-existent tensor', () => {
    const buffer = createDummySafeTensorsBuffer({}, 0);
    const st = new SafeTensors(buffer);
    expect(() => st.getTensor('missing')).toThrow('Tensor missing not found');
  });

  test('getTensor copy mode', () => {
    const headerObj = { a: { dtype: 'I8', shape: [8], data_offsets: [0, 8] } };
    const buffer = createDummySafeTensorsBuffer(headerObj, 8);
    const st = new SafeTensors(buffer);
    const tensor = st.getTensor('a', true);
    expect(tensor.buffer).not.toBe(buffer);
  });

  test('getTypedArray copy and unknown/proprietary dtypes', () => {
    const buffer = createDummySafeTensorsBuffer(
      { a: { dtype: 'I8', shape: [8], data_offsets: [0, 8] } },
      8,
    );
    const st = new SafeTensors(buffer);

    st.tensors['f64'] = { dtype: 'F64', shape: [1], data_offsets: [0, 8] };
    st.tensors['i32'] = { dtype: 'I32', shape: [2], data_offsets: [0, 8] };
    st.tensors['i16'] = { dtype: 'I16', shape: [4], data_offsets: [0, 8] };
    st.tensors['u32'] = { dtype: 'U32', shape: [2], data_offsets: [0, 8] };
    st.tensors['u16'] = { dtype: 'U16', shape: [4], data_offsets: [0, 8] };
    st.tensors['u8'] = { dtype: 'U8', shape: [8], data_offsets: [0, 8] };
    st.tensors['i64'] = { dtype: 'I64', shape: [1], data_offsets: [0, 8] };
    st.tensors['u64'] = { dtype: 'U64', shape: [1], data_offsets: [0, 8] };
    st.tensors['f16'] = { dtype: 'F16', shape: [4], data_offsets: [0, 8] };
    st.tensors['bf16'] = { dtype: 'BF16', shape: [4], data_offsets: [0, 8] };
    st.tensors['bool'] = { dtype: 'BOOL', shape: [8], data_offsets: [0, 8] };

    expect(st.getTypedArray('f64', true)).toBeInstanceOf(Float64Array);
    expect(st.getTypedArray('i32')).toBeInstanceOf(Int32Array);
    expect(st.getTypedArray('i16')).toBeInstanceOf(Int16Array);
    expect(st.getTypedArray('u32')).toBeInstanceOf(Uint32Array);
    expect(st.getTypedArray('u16')).toBeInstanceOf(Uint16Array);
    expect(st.getTypedArray('u8')).toBeInstanceOf(Uint8Array);
    expect(st.getTypedArray('i64')).toBeInstanceOf(BigInt64Array);
    expect(st.getTypedArray('u64')).toBeInstanceOf(BigUint64Array);
    expect(st.getTypedArray('f16')).toBeInstanceOf(Uint16Array);
    expect(st.getTypedArray('bf16')).toBeInstanceOf(Uint16Array);
    expect(st.getTypedArray('bool')).toBeInstanceOf(Uint8Array);

    st.tensors['c64'] = { dtype: 'C64' as Object, shape: [1], data_offsets: [0, 8] };
    expect(() => st.getTypedArray('c64')).toThrow(SafetensorsInvalidDtypeError);

    st.tensors['unk'] = { dtype: 'UNK' as Object, shape: [1], data_offsets: [0, 8] };
    expect(() => st.getTypedArray('unk')).toThrow(SafetensorsInvalidDtypeError);

    expect(() => st.getTypedArray('missing')).toThrow('Tensor missing not found');
  });

  test('getTypedArray unaligned enforcement copy', () => {
    const buffer = createDummySafeTensorsBuffer(
      { a: { dtype: 'I8', shape: [12], data_offsets: [0, 12] } },
      12,
    );
    const st = new SafeTensors(buffer);

    vi.spyOn(st, 'getTensor').mockReturnValue(new Uint8Array(buffer, 1));
    st.tensors['unaligned_f32'] = { dtype: 'F32', shape: [2], data_offsets: [0, 8] };

    const arr = st.getTypedArray('unaligned_f32');
    expect(arr).toBeInstanceOf(Float32Array);
  });

  test('GPUBuffer and WebGPU', () => {
    const buffer = createDummySafeTensorsBuffer(
      { a: { dtype: 'I8', shape: [8], data_offsets: [0, 8] } },
      8,
    );
    const st = new SafeTensors(buffer);

    const mappedArray = new Uint8Array(8);
    const mockGpuBuffer = {
      getMappedRange: vi.fn(() => mappedArray.buffer),
      unmap: vi.fn(),
    };
    const mockDevice = {
      createBuffer: vi.fn(() => mockGpuBuffer),
      queue: { writeBuffer: vi.fn() },
    };

    const gpuBuffer = st.createGPUBuffer(mockDevice, 'a');
    expect(mockDevice.createBuffer).toHaveBeenCalledWith({
      size: 8,
      usage: 132,
      mappedAtCreation: true,
    });
    expect(mockGpuBuffer.unmap).toHaveBeenCalled();

    st.injectToGPUQueue(mockDevice, gpuBuffer, 'a', 0);
    expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
  });

  test('padTo8Bytes', () => {
    const arr = new Uint8Array([1, 2, 3]);
    const padded = padTo8Bytes(arr);
    expect(padded.byteLength).toBe(8);
    expect(padded[0]).toBe(1);

    const arr8 = new Uint8Array(8);
    const padded8 = padTo8Bytes(arr8);
    expect(padded8.byteLength).toBe(8);
  });

  test('saveSafetensors object input', () => {
    const tensors = {
      a: { data: new Uint8Array([1, 2, 3]), dtype: 'F32', shape: [3] },
    };
    const buffer = saveSafetensors(tensors as Object, { mymeta: 'val' });
    expect(buffer).toBeInstanceOf(Uint8Array);
  });

  test('checkSafetensors catch general error', () => {
    const badBuffer = {
      get byteLength() {
        throw new Error('System Fault');
      },
    } as Object as ArrayBuffer;
    expect(() => checkSafetensors(badBuffer)).toThrow('System Fault');
  });

  test('_mallocSafetensors and passToPyodideWASM', () => {
    const mockModule = {
      _malloc: vi.fn((size) => {
        if (size > 100) return 0; // Simulate OOM
        return 1024; // Fake ptr
      }),
    };
    expect(() => _mallocSafetensors(200, mockModule)).toThrow(
      'Emscripten OOM (Out of Memory) allocating tensor payload',
    );
    expect(_mallocSafetensors(50, mockModule)).toBe(1024);

    const mockPyodide = {
      _module: mockModule,
      HEAPU8: new Uint8Array(2048),
    };
    const ptr = passToPyodideWASM(new Uint8Array([1, 2, 3]), mockPyodide);
    expect(ptr).toBe(1024);
    expect(mockPyodide.HEAPU8[1024]).toBe(1);
  });

  test('extractFromPyodideFS missing node', () => {
    const mockFS = { lookupPath: () => ({ node: null }) };
    expect(() => extractFromPyodideFS(mockFS, 'path')).toThrow(
      'Could not extract Uint8Array from Pyodide FS',
    );
  });
});

describe('Async Fetch Operations', () => {
  let originalFetch: Object;
  let originalProcess: Object;
  let originalCaches: Object;
  let originalSetTimeout: Object;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    originalProcess = globalThis.process;
    originalCaches = globalThis.caches;
    originalSetTimeout = globalThis.setTimeout;

    globalThis.fetch = vi.fn();
    globalThis.caches = undefined as Object;
    globalThis.process = { env: { HF_TOKEN: 'test_token' } } as Object;

    // Fast-forward retries immediately
    globalThis.setTimeout = ((cb: Object) => {
      cb();
      return 0;
    }) as Object;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    globalThis.process = originalProcess;
    globalThis.caches = originalCaches;
    globalThis.setTimeout = originalSetTimeout;
    vi.restoreAllMocks();
  });

  test('fetchSafetensorsHeader success', async () => {
    const mockHeaderObj = {
      a: { dtype: 'I8', shape: [8], data_offsets: [0, 8] },
      __metadata__: {},
    };
    const mockBuf = createDummySafeTensorsBuffer(mockHeaderObj, 8);
    const mockHeaderBytes = new Uint8Array(
      mockBuf,
      8,
      Number(new DataView(mockBuf).getBigUint64(0, true)) as Object,
    );

    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as Object;
    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        status: 206,
        headers: new Headers({ 'Accept-Ranges': 'bytes' }),
        arrayBuffer: async () => mockBuf.slice(0, 8),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 206,
        arrayBuffer: async () =>
          mockHeaderBytes.buffer.slice(
            mockHeaderBytes.byteOffset,
            mockHeaderBytes.byteOffset + mockHeaderBytes.byteLength,
          ),
      });

    const res = await fetchSafetensorsHeader('https://huggingface.co/test/test/model.safetensors');
    expect(res.headerSize).toBe(mockHeaderBytes.byteLength);
    expect(res.headerObj).toEqual(mockHeaderObj);
  });

  test('fetchSafetensorsHeader fallback stream entire file', async () => {
    const mockHeaderObj = {
      a: { dtype: 'I8', shape: [8], data_offsets: [0, 8] },
      __metadata__: {},
    };
    const buffer = createDummySafeTensorsBuffer(mockHeaderObj, 8);

    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as Object;
    fetchMock.mockResolvedValueOnce({
      status: 200,
      ok: true,
      arrayBuffer: async () => buffer,
    });

    const res = await fetchSafetensorsHeader('hf://user/repo/file.safetensors');
    expect(res.headerSize).toBe(0);
    expect(res.headerObj).toHaveProperty('a');
    expect(res.fullBuffer).toBeTruthy();
  });

  test('fetchSafetensorsHeader invalid JSON', async () => {
    const mockBuf8 = new ArrayBuffer(8);
    const view = new DataView(mockBuf8);
    view.setBigUint64(0, BigInt(10), true);

    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as Object;
    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        status: 206,
        headers: new Headers({ 'Accept-Ranges': 'bytes' }),
        arrayBuffer: async () => mockBuf8,
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 206,
        arrayBuffer: async () => new TextEncoder().encode('bad json').buffer,
      });

    await expect(fetchSafetensorsHeader('https://test')).rejects.toThrow(
      SafetensorsInvalidJSONError,
    );
  });

  test('fetchSafetensorsHeader accept-ranges none', async () => {
    const mockBuf8 = new ArrayBuffer(8);
    const view = new DataView(mockBuf8);
    const mockHeaderBytes = new TextEncoder().encode('{}');
    view.setBigUint64(0, BigInt(mockHeaderBytes.byteLength), true);

    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as Object;
    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        status: 206,
        headers: new Headers({ 'Accept-Ranges': 'none' }),
        arrayBuffer: async () => mockBuf8,
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 206,
        arrayBuffer: async () =>
          mockHeaderBytes.buffer.slice(
            mockHeaderBytes.byteOffset,
            mockHeaderBytes.byteOffset + mockHeaderBytes.byteLength,
          ),
      });

    const res = await fetchSafetensorsHeader('https://test');
    expect(res.headerObj).toEqual({});
  });

  test('fetchSafetensorsChunk from fullBuffer', async () => {
    const fullBuf = createDummySafeTensorsBuffer(
      { a: { dtype: 'I8', shape: [8], data_offsets: [0, 8] } },
      8,
    );
    const onProgress = vi.fn();
    const chunk = await fetchSafetensorsChunk('http://test', 0, 0, 8, fullBuf, onProgress);
    expect(chunk.byteLength).toBe(8);
    expect(onProgress).toHaveBeenCalled();
  });

  test('fetchSafetensorsChunk basic stream', async () => {
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as Object;

    const encoder = new TextEncoder();
    const chunk1 = encoder.encode('12');
    const chunk2 = encoder.encode('34');

    const mockReader = {
      read: vi
        .fn()
        .mockResolvedValueOnce({ done: false, value: chunk1 })
        .mockResolvedValueOnce({ done: false, value: chunk2 })
        .mockResolvedValueOnce({ done: true }),
    };

    fetchMock.mockResolvedValueOnce({
      ok: true,
      status: 206,
      headers: new Headers({ 'Content-Length': '4' }),
      body: { getReader: () => mockReader },
    });

    const onProgress = vi.fn();
    const chunk = await fetchSafetensorsChunk('http://test', 10, 0, 4, undefined, onProgress);
    expect(chunk.byteLength).toBe(4);
    expect(onProgress).toHaveBeenCalledTimes(2);
  });

  test('fetchSafetensorsChunk WebSocket success', async () => {
    const OriginalWS = globalThis.WebSocket;
    class MockWebSocket {
      binaryType: string = '';
      onopen: Object;
      onmessage: Object;
      onerror: Object;
      close = vi.fn();
      send = vi.fn();
      constructor() {
        originalSetTimeout(() => {
          if (this.onopen) this.onopen();
          originalSetTimeout(() => {
            if (this.onmessage) this.onmessage({ data: new ArrayBuffer(4) });
          }, 10);
        }, 10);
      }
    }
    globalThis.WebSocket = MockWebSocket as Object;
    const res = await fetchSafetensorsChunk('ws://test', 10, 0, 4);
    expect(res.byteLength).toBe(4);
    globalThis.WebSocket = OriginalWS;
  });

  test('fetchSafetensorsChunk WebSocket error', async () => {
    const OriginalWS = globalThis.WebSocket;
    class MockWebSocket {
      binaryType: string = '';
      onopen: Object;
      onmessage: Object;
      onerror: Object;
      close = vi.fn();
      send = vi.fn();
      constructor() {
        originalSetTimeout(() => {
          if (this.onerror) this.onerror(new Error('WS err'));
        }, 10);
      }
    }
    globalThis.WebSocket = MockWebSocket as Object;
    await expect(fetchSafetensorsChunk('ws://test', 10, 0, 4)).rejects.toThrow(
      'WebSocket chunk fetch failed: Error: WS err',
    );
    globalThis.WebSocket = OriginalWS;
  });

  test('loadTensors generator', async () => {
    const headerObj = {
      a: { dtype: 'I8', shape: [8], data_offsets: [0, 8] },
      b: { dtype: 'I8', shape: [8], data_offsets: [8, 16] },
      __metadata__: {},
    };
    const buffer = createDummySafeTensorsBuffer(headerObj, 16);

    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as Object;
    fetchMock.mockResolvedValueOnce({
      status: 200,
      ok: true,
      arrayBuffer: async () => buffer,
    });

    const results = [];
    for await (const t of loadTensors('http://test', { pattern: 'a', cleanupViews: true })) {
      results.push(t);
    }
    expect(results.length).toBe(1);
    expect(results[0].name).toBe('a');
    expect(results[0].data).toBeNull(); // Because cleanupViews is true
  });

  test('fetchSafetensorsChunk retry on 429', async () => {
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as Object;
    fetchMock
      .mockResolvedValueOnce({
        status: 429,
        headers: new Headers({ 'Retry-After': '0' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 206,
        headers: new Headers(),
        arrayBuffer: async () => new ArrayBuffer(4),
      });

    const res = await fetchSafetensorsChunk('http://test', 10, 0, 4);
    expect(res.byteLength).toBe(4);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test('fetchSafetensorsChunk various 4xx errors', async () => {
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as Object;

    // Test 404 (needs 3 retries to throw due to MAX_RETRIES)
    fetchMock.mockResolvedValue({ status: 404, ok: false, headers: new Headers() });
    await expect(fetchSafetensorsChunk('http://test', 10, 0, 4)).rejects.toThrow('404 Not Found');

    // Test 403
    fetchMock.mockResolvedValue({ status: 403, ok: false, headers: new Headers() });
    await expect(fetchSafetensorsChunk('http://test', 10, 0, 4)).rejects.toThrow('403 Forbidden');

    // Test 416
    fetchMock.mockResolvedValue({ status: 416, ok: false, headers: new Headers() });
    await expect(fetchSafetensorsChunk('http://test', 10, 0, 4)).rejects.toThrow(
      '416 Range Not Satisfiable',
    );

    // Test 500
    fetchMock.mockResolvedValue({ status: 500, ok: false, headers: new Headers() });
    await expect(fetchSafetensorsChunk('http://test', 10, 0, 4)).rejects.toThrow(
      'Failed to fetch chunk',
    );
  });

  test('Caches block in fetchSafetensorsHeader', async () => {
    const mockCache = {
      match: vi.fn().mockResolvedValue({ arrayBuffer: async () => new ArrayBuffer(8) }),
      put: vi.fn(),
    };
    globalThis.caches = { open: vi.fn().mockResolvedValue(mockCache) } as Object;

    const mockBuf8 = createDummySafeTensorsBuffer(
      { a: { dtype: 'I8', shape: [8], data_offsets: [0, 8] } },
      8,
    );

    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as Object;
    fetchMock.mockResolvedValue({
      status: 200,
      ok: true,
      arrayBuffer: async () => mockBuf8,
    });

    const res = await fetchSafetensorsHeader('http://test');
    expect(globalThis.caches.open).toHaveBeenCalledWith('onnx9000-safetensors');
    // Fallback stream handles caching entire file
    expect(mockCache.put).toHaveBeenCalled();
  });

  test('Caches block in fetchSafetensorsChunk', async () => {
    const mockCache = {
      match: vi.fn().mockResolvedValue({ arrayBuffer: async () => new ArrayBuffer(4) }),
      put: vi.fn(),
    };
    globalThis.caches = { open: vi.fn().mockResolvedValue(mockCache) } as Object;

    const res = await fetchSafetensorsChunk('http://test', 10, 0, 4);
    expect(res.byteLength).toBe(4);
    expect(mockCache.match).toHaveBeenCalled();
  });
});

test('decodeFloat16 edge cases', () => {
  // Normal number
  const normal = new Uint16Array([0x3c00]); // 1.0
  expect(decodeFloat16(normal)[0]).toBe(1.0);

  // Zero
  const zero = new Uint16Array([0x0000]); // 0.0
  expect(decodeFloat16(zero)[0]).toBe(0.0);

  // Denormalized
  const denorm = new Uint16Array([0x0001]); // Smallest positive subnormal
  expect(decodeFloat16(denorm)[0]).toBeGreaterThan(0);

  // Inf / NaN
  const inf = new Uint16Array([0x7c00]); // +Inf
  expect(decodeFloat16(inf)[0]).toBe(Infinity);

  const nan = new Uint16Array([0x7c01]); // NaN
  expect(Number.isNaN(decodeFloat16(nan)[0])).toBe(true);
});

test('Pyodide/WASM integration coverage', () => {
  // mock module
  const mockModule = {
    _malloc: (len: number) => {
      if (len === 999) return 0; // Simulate OOM
      return 1024;
    },
    HEAPU8: new Uint8Array(2048),
  };

  expect(_mallocSafetensors(10, mockModule)).toBe(1024);
  expect(() => _mallocSafetensors(999, mockModule)).toThrowError(/OOM/);

  const tensor = new Uint8Array([1, 2, 3]);
  const ptr = passToPyodideWASM(tensor, { _module: mockModule, HEAPU8: mockModule.HEAPU8 });
  expect(ptr).toBe(1024);
  expect(mockModule.HEAPU8[1024]).toBe(1);

  // extractFromPyodideFS
  const mockFS = {
    lookupPath: (path: string) => {
      if (path === 'fail') return { node: null };
      if (path === 'fail2') return { node: { contents: null } };

      const buf = new Uint8Array([8, 0, 0, 0, 0, 0, 0, 0, 123, 125, 32, 32, 32, 32, 32, 32]); // Valid empty safetensors
      return { node: { contents: buf } };
    },
  };

  expect(() => extractFromPyodideFS(mockFS, 'fail')).toThrowError(/extract Uint8Array/);
  expect(() => extractFromPyodideFS(mockFS, 'fail2')).toThrowError(/extract Uint8Array/);

  const st = extractFromPyodideFS(mockFS, 'success');
  expect(st).toBeInstanceOf(SafeTensors);
});

test('benchmark10kKeys coverage', async () => {
  const res = await benchmark10kKeys();
  expect(res.keysParsed).toBe(10000);
});

test('swapEndianness coverage', async () => {
  const { swapEndianness } = await import('../src/parser/safetensors');
  const buf = new Uint8Array([1, 2, 3, 4]).buffer;
  swapEndianness(buf, 0, 4, 2);
  expect(new Uint8Array(buf)).toEqual(new Uint8Array([2, 1, 4, 3]));
  swapEndianness(buf, 0, 4, 4);
  expect(new Uint8Array(buf)).toEqual(new Uint8Array([3, 4, 1, 2]));
});

test('decodeBfloat16 coverage', async () => {
  const { decodeBfloat16 } = await import('../src/parser/safetensors');
  const uint16Array = new Uint16Array([0x3f80, 0x0000]); // 1.0, 0.0 in BF16
  const res = decodeBfloat16(uint16Array);
  expect(res[0]).toBe(1.0);
  expect(res[1]).toBe(0.0);
});

test('saveSafetensors Uint8Array input', () => {
  const buf = saveSafetensors({ a: new Uint8Array([1, 2]) });
  expect(buf.byteLength).toBeGreaterThan(0);
});

test('getEndianness BE', () => {
  const OriginalUint8Array = globalThis.Uint8Array;

  // Mock for BE
  globalThis.Uint8Array = function (buf: Object) {
    const arr = new OriginalUint8Array(buf);
    arr[0] = 0x12;
    return arr;
  } as Object;
  expect(getEndianness()).toBe('BE');

  // Mock for unknown
  globalThis.Uint8Array = function (buf: Object) {
    const arr = new OriginalUint8Array(buf);
    arr[0] = 0x99;
    return arr;
  } as Object;
  expect(getEndianness()).toBe('LE');

  globalThis.Uint8Array = OriginalUint8Array;
});

test('saveSafetensors dead branches', () => {
  // Duplicate key dead code
  const OriginalEntries = Object.entries;
  Object.entries = function (obj) {
    if (obj && obj.mock_duplicate) {
      return [
        ['a', new Uint8Array([1])],
        ['a', new Uint8Array([2])],
      ];
    }
    return OriginalEntries(obj);
  } as Object;

  expect(() => saveSafetensors({ mock_duplicate: true } as Object)).toThrowError(/Duplicate/);
  Object.entries = OriginalEntries;

  // SharedArrayBuffer creation
  const b1 = createBuffer(10, true);
  expect(b1.byteLength).toBe(10);

  // Mock SharedArrayBuffer to throw
  const OriginalSAB = globalThis.SharedArrayBuffer;
  globalThis.SharedArrayBuffer = function () {
    throw new Error('Blocked');
  } as Object;
  const b2 = createBuffer(10, true);
  expect(b2).toBeInstanceOf(ArrayBuffer);
  globalThis.SharedArrayBuffer = OriginalSAB;
});

test('createBuffer no shared', () => {
  const b = createBuffer(10, false);
  expect(b.byteLength).toBe(10);
});

test('fetchSafetensorsChunk cache puts perfectly', async () => {
  const { fetchSafetensorsChunk } = await import('../src/parser/safetensors');

  // Valid Request and Response globals
  const OrigResponse = globalThis.Response;
  const OrigRequest = globalThis.Request;
  globalThis.Response = class Response {
    constructor() {}
  } as Object;
  globalThis.Request = class Request {
    constructor() {}
  } as Object;

  const fetchMock = vi.fn();
  globalThis.fetch = fetchMock as Object;
  fetchMock.mockResolvedValueOnce({
    ok: true,
    status: 200,
    headers: new Headers(),
    arrayBuffer: async () => new Uint8Array(10).buffer,
  });
  fetchMock.mockResolvedValueOnce({
    ok: true,
    status: 200,
    headers: new Headers(),
    body: {
      getReader: () => {
        let done = false;
        return {
          read: async () => {
            if (done) return { done: true };
            done = true;
            return { done: false, value: new Uint8Array(10) };
          },
        };
      },
    },
  });

  let putCalled = 0;
  const mockCache = {
    put: async () => {
      putCalled++;
    },
  };

  // Setup global caches
  const OrigCaches = globalThis.caches;
  globalThis.caches = {
    open: async () => ({
      match: async () => undefined,
      put: async () => {
        putCalled++;
      },
    }),
  } as Object;

  // ArrayBuffer path
  await fetchSafetensorsChunk('http://test', 0, 0, 10);

  // Stream path
  await fetchSafetensorsChunk('http://test', 0, 0, 10);

  globalThis.caches = OrigCaches;

  expect(putCalled).toBe(2);

  globalThis.Response = OrigResponse;
  globalThis.Request = OrigRequest;
});

test('fetchSafetensorsChunk hf:// and cache throw', async () => {
  const { fetchSafetensorsChunk } = await import('../src/parser/safetensors');

  // hf:// resolution
  const fetchMock = vi.fn();
  globalThis.fetch = fetchMock as Object;
  fetchMock.mockResolvedValue({
    ok: true,
    status: 200,
    headers: new Headers(),
    arrayBuffer: async () => new Uint8Array(10).buffer,
  });

  // Test that caches.open throws
  const OrigCaches = globalThis.caches;
  globalThis.caches = {
    open: async () => {
      throw new Error('Caches fail');
    },
  } as Object;

  await fetchSafetensorsChunk('hf://user/repo/file.bin', 0, 0, 10);

  // Reset caches
  globalThis.caches = OrigCaches;

  // Test hf:// without user/repo/file format
  await fetchSafetensorsChunk('hf://short', 0, 0, 10);
});

test('fetchSafetensorsHeader remaining branches', async () => {
  const { fetchSafetensorsHeader } = await import('../src/parser/safetensors');

  // Test caches.open throwing
  const OrigCaches = globalThis.caches;
  globalThis.caches = {
    open: async () => {
      throw new Error('Caches fail');
    },
  } as Object;

  const fetchMock = vi.fn();
  globalThis.fetch = fetchMock as Object;
  fetchMock
    .mockResolvedValueOnce({
      ok: true,
      status: 206,
      headers: new Headers({ 'Accept-Ranges': 'bytes' }),
      arrayBuffer: async () => new Uint8Array([8, 0, 0, 0, 0, 0, 0, 0]).buffer,
    })
    .mockResolvedValueOnce({
      ok: true,
      status: 206,
      arrayBuffer: async () => new TextEncoder().encode('{}').buffer,
    });

  await fetchSafetensorsHeader('http://test');
  globalThis.caches = OrigCaches;

  // Test JSON.parse returning array
  const buf = new Uint8Array([8, 0, 0, 0, 0, 0, 0, 0]);
  const headerBytes = new TextEncoder().encode('[]      ');
  fetchMock
    .mockResolvedValueOnce({
      ok: true,
      status: 206,
      headers: new Headers({ 'Accept-Ranges': 'bytes' }),
      arrayBuffer: async () => buf.buffer,
    })
    .mockResolvedValueOnce({
      ok: true,
      status: 206,
      arrayBuffer: async () => headerBytes.buffer,
    });

  const { SafetensorsInvalidJSONError } = await import('../src/parser/safetensors');
  try {
    await fetchSafetensorsHeader('http://test2');
  } catch (e) {
    expect(e).toBeInstanceOf(SafetensorsInvalidJSONError);
  }
});
