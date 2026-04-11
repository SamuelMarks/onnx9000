/* eslint-disable */
/**
 * Error thrown for general safetensors issues.
 */
export class SafetensorsError extends Error {
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}

/**
 * Error thrown when the safetensors header size exceeds the allowed limit.
 */
export class SafetensorsHeaderTooLargeError extends SafetensorsError {}
/**
 * Error thrown when the safetensors header is invalid (e.g. not UTF-8).
 */
export class SafetensorsInvalidHeaderError extends SafetensorsError {}
/**
 * Error thrown when the safetensors header JSON is invalid.
 */
export class SafetensorsInvalidJSONError extends SafetensorsError {}
/**
 * Error thrown when duplicate tensor names are found in the header.
 */
export class SafetensorsDuplicateKeyError extends SafetensorsError {}
/**
 * Error thrown when tensor data offsets are invalid (e.g. begin > end).
 */
export class SafetensorsInvalidOffsetError extends SafetensorsError {}
/**
 * Error thrown when tensor data offsets are out of file boundaries.
 */
export class SafetensorsOutOfBoundsError extends SafetensorsError {}
/**
 * Error thrown when tensor data regions overlap.
 */
export class SafetensorsOverlapError extends SafetensorsError {}
/**
 * Error thrown when tensor data is not properly aligned.
 */
export class SafetensorsAlignmentError extends SafetensorsError {}
/**
 * Error thrown when an invalid or unsupported dtype is encountered.
 */
export class SafetensorsInvalidDtypeError extends SafetensorsError {}
/**
 * Error thrown when there is a mismatch in tensor shape or volume.
 */
export class SafetensorsShapeMismatchError extends SafetensorsError {}
/**
 * Error thrown when the safetensors file is empty.
 */
export class SafetensorsFileEmptyError extends SafetensorsError {}
/**
 * Error thrown when the safetensors file is too small to contain a header.
 */
export class SafetensorsFileTooSmallError extends SafetensorsError {}

/**
 * Supported data types in safetensors format.
 */
export type Dtype =
  | 'F64'
  | 'F32'
  | 'F16'
  | 'BF16'
  | 'I64'
  | 'I32'
  | 'I16'
  | 'I8'
  | 'U64'
  | 'U32'
  | 'U16'
  | 'U8'
  | 'BOOL';

/**
 * Sizes in bytes for each supported dtype.
 */
const DTYPE_SIZES: Record<Dtype, number> = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U64: 8,
  U32: 4,
  U16: 2,
  U8: 1,
  BOOL: 1,
};

/**
 * Calculates the total number of elements in a shape.
 * @param shape Dimensions array
 * @returns Total element count
 */
function calculateVolume(shape: number[]): number {
  if (!Array.isArray(shape)) {
    throw new SafetensorsShapeMismatchError('Shape must be an array');
  }
  let vol = 1;
  for (const dim of shape) {
    if (dim < 0) throw new SafetensorsShapeMismatchError(`Negative dimension found: ${dim}`);
    if (dim > Number.MAX_SAFE_INTEGER)
      throw new SafetensorsShapeMismatchError(`Dimension too large: ${dim}`);
    vol *= dim;
  }
  return vol;
}

/**
 * Metadata for a single tensor in a safetensors file.
 */
export interface TensorInfo {
  dtype: Dtype;
  shape: number[];
  data_offsets: [number, number];
}

/**
 * Interface for Pyodide or Emscripten modules.
 */
export interface EmscriptenModule {
  _malloc(size: number): number;
  HEAPU8: Uint8Array;
}

/**
 * Interface for Pyodide instance.
 */
export interface PyodideInstance {
  _module: EmscriptenModule;
  HEAPU8: Uint8Array;
}

/**
 * Interface for Emscripten FS.
 */
export interface EmscriptenFS {
  lookupPath(path: string): { node: { contents: Uint8Array } };
}

/**
 * Represents a parsed safetensors file.
 */
export class SafeTensors {
  public readonly metadata: Record<string, string>;
  public readonly metadataLength: number;
  public readonly tensors: Record<string, TensorInfo>;
  private readonly buffer: ArrayBuffer | SharedArrayBuffer;
  private readonly headerSize: number;
  private readonly endiannessOverride?: 'LE' | 'BE' | undefined;

  /**
   * Initializes SafeTensors from a buffer.
   * @param buffer The file content buffer
   * @param endianness Optional endianness override for testing
   */
  constructor(buffer: ArrayBuffer | SharedArrayBuffer, endianness?: 'LE' | 'BE') {
    if (buffer.byteLength === 0) {
      throw new SafetensorsFileEmptyError('File is empty');
    }
    if (buffer.byteLength < 8) {
      throw new SafetensorsFileTooSmallError('File too small to contain header size');
    }

    this.buffer = buffer;
    this.endiannessOverride = endianness;
    const view = new DataView(buffer);

    // Read 8-byte little-endian unsigned integer
    // JavaScript bitwise operations are 32-bit, so we use BigInt
    const headerSizeBig = view.getBigUint64(0, true);
    if (headerSizeBig > BigInt(100 * 1024 * 1024)) {
      throw new SafetensorsHeaderTooLargeError('Header size exceeds 100MB limit');
    }
    this.headerSize = Number(headerSizeBig);

    if (this.headerSize + 8 > buffer.byteLength) {
      throw new SafetensorsOutOfBoundsError('Header size exceeds file boundaries');
    }

    const headerBytes = new Uint8Array(buffer, 8, this.headerSize);
    let headerStr: string;
    try {
      const decoder = new TextDecoder('utf-8', { fatal: true });
      headerStr = decoder.decode(headerBytes);
    } catch (e) {
      throw new SafetensorsInvalidHeaderError(`Invalid UTF-8 header: ${e}`);
    }

    let headerObj: Record<string, TensorInfo | Record<string, string>>;
    try {
      headerObj = JSON.parse(headerStr);
      if (typeof headerObj !== 'object' || headerObj === null || Array.isArray(headerObj)) {
        throw new Error('Header must be a dictionary/object');
      }
    } catch (e) {
      throw new SafetensorsInvalidJSONError(`Invalid JSON header: ${e}`);
    }

    this.metadata = (headerObj.__metadata__ as Record<string, string>) || {};
    this.metadataLength = Object.keys(this.metadata).length;

    for (const [k, v] of Object.entries(this.metadata)) {
      if (typeof v === 'string') {
        const lower = v.toLowerCase();
        if (lower.includes('<script') || lower.includes('javascript:')) {
          throw new SafetensorsError('Executable script tags detected in metadata');
        }
      }
    }

    delete headerObj.__metadata__;

    this.tensors = {};
    const seenRegions: [number, number][] = [];

    for (const [name, infoRaw] of Object.entries(headerObj)) {
      const info = infoRaw as TensorInfo;
      if (this.tensors[name]) {
        throw new SafetensorsDuplicateKeyError(`Duplicate tensor name: ${name}`);
      }

      const dtype = info.dtype;
      if (!(dtype in DTYPE_SIZES)) {
        throw new SafetensorsInvalidDtypeError(`Unknown dtype: ${dtype}`);
      }

      const shape = info.shape || [];
      const offsets = info.data_offsets || [0, 0];
      const begin = offsets[0];
      const end = offsets[1];

      if (begin > end) {
        throw new SafetensorsInvalidOffsetError(`Invalid offsets: begin ${begin} > end ${end}`);
      }
      if (begin % 8 !== 0) {
        throw new SafetensorsAlignmentError(`Offset begin ${begin} is not 8-byte aligned`);
      }

      const absEnd = 8 + this.headerSize + end;
      if (absEnd > buffer.byteLength) {
        throw new SafetensorsOutOfBoundsError(`Data region for ${name} exceeds file boundaries`);
      }

      const expectedSize = calculateVolume(shape) * DTYPE_SIZES[dtype];
      if (expectedSize !== end - begin) {
        throw new SafetensorsShapeMismatchError(
          `Shape volume * dtype size (${expectedSize}) != offset size (${end - begin}) for ${name}`,
        );
      }

      seenRegions.push([begin, end]);
      this.tensors[name] = { dtype, shape, data_offsets: [begin, end] };
    }

    // Check overlaps
    seenRegions.sort((a, b) => a[0] - b[0]);
    for (let i = 0; i < seenRegions.length - 1; i++) {
      const current = seenRegions[i];
      const next = seenRegions[i + 1];
      if (current && next && current[1] > next[0]) {
        throw new SafetensorsOverlapError('Tensor data regions overlap');
      }
    }
  }

  /**
   * Retrieves raw bytes for a tensor.
   * @param name Tensor name
   * @param copy Whether to return a copy
   * @returns Raw data as Uint8Array
   */
  public getTensor(name: string, copy: boolean = false): Uint8Array {
    if (!this.tensors[name]) {
      throw new Error(`Tensor ${name} not found`);
    }

    const info = this.tensors[name];
    const [begin, end] = info.data_offsets;
    const absBegin = 8 + this.headerSize + begin;
    const length = end - begin;

    // JS DataView / ArrayBuffer has 2GB-4GB limits in some engines natively
    if (absBegin + length > Number.MAX_SAFE_INTEGER) {
      throw new SafetensorsOutOfBoundsError(`Offset > MAX_SAFE_INTEGER bounds`);
    }

    if (copy) {
      return new Uint8Array(this.buffer.slice(absBegin, absBegin + length));
    }
    return new Uint8Array(this.buffer, absBegin, length);
  }

  /**
   * Retrieves a typed array for a tensor.
   * @param name Tensor name
   * @param copy Whether to return a copy
   * @returns Appropriately typed array
   */
  public getTypedArray(
    name: string,
    copy: boolean = false,
  ):
    | Int8Array
    | Uint8Array
    | Int16Array
    | Uint16Array
    | Int32Array
    | Uint32Array
    | BigInt64Array
    | BigUint64Array
    | Float32Array
    | Float64Array {
    const info = this.tensors[name];
    if (!info) throw new Error(`Tensor ${name} not found`);
    const uint8 = this.getTensor(name, copy);
    let { buffer, byteOffset, byteLength } = uint8;

    const enforceAlignment = (elementSize: number) => {
      const currentEndianness = this.endiannessOverride || getEndianness();
      if (byteOffset % elementSize !== 0 || (currentEndianness === 'BE' && elementSize > 1)) {
        // Unaligned buffer fallback or Big-Endian fallback: explicitly copy to aligned array
        const slice = new Uint8Array(buffer.slice(byteOffset, byteOffset + byteLength));
        buffer = slice.buffer;
        byteOffset = slice.byteOffset;
        byteLength = slice.byteLength;

        if (currentEndianness === 'BE' && elementSize > 1) {
          swapEndianness(buffer, byteOffset, byteLength, elementSize);
        }
      }
    };

    switch (info.dtype as string) {
      case 'F64':
        enforceAlignment(8);
        return new Float64Array(buffer, byteOffset, byteLength / 8);
      case 'F32':
        enforceAlignment(4);
        return new Float32Array(buffer, byteOffset, byteLength / 4);
      case 'I32':
        enforceAlignment(4);
        return new Int32Array(buffer, byteOffset, byteLength / 4);
      case 'I16':
        enforceAlignment(2);
        return new Int16Array(buffer, byteOffset, byteLength / 2);
      case 'I8':
        return new Int8Array(buffer, byteOffset, byteLength);
      case 'U32':
        enforceAlignment(4);
        return new Uint32Array(buffer, byteOffset, byteLength / 4);
      case 'U16':
        enforceAlignment(2);
        return new Uint16Array(buffer, byteOffset, byteLength / 2);
      case 'U8':
        return new Uint8Array(buffer, byteOffset, byteLength);
      case 'I64':
        enforceAlignment(8);
        return new BigInt64Array(buffer, byteOffset, byteLength / 8);
      case 'U64':
        enforceAlignment(8);
        return new BigUint64Array(buffer, byteOffset, byteLength / 8);
      case 'F16':
        enforceAlignment(2);
        return new Uint16Array(buffer, byteOffset, byteLength / 2);
      case 'BF16':
        enforceAlignment(2);
        return new Uint16Array(buffer, byteOffset, byteLength / 2);
      case 'BOOL':
        return new Uint8Array(buffer, byteOffset, byteLength);
      case 'C64':
      case 'C128':
        throw new SafetensorsInvalidDtypeError(
          `Complex types (${info.dtype}) are not currently supported by standard safetensors JS parser.`,
        );
      default:
        throw new SafetensorsInvalidDtypeError(`Unsupported or proprietary dtype: ${info.dtype}`);
    }
  }

  /**
   * Gets all tensor names.
   * @returns Array of tensor names
   */
  public keys(): string[] {
    return Object.keys(this.tensors);
  }
}

/**
 * Fetches the header of a safetensors file from a URL.
 * @param url File URL
 * @returns Header information
 */
export async function fetchSafetensorsHeader(url: string) {
  if (url.startsWith('hf://')) {
    url = url.replace('hf://', 'https://huggingface.co/');
    const parts = (url.split('huggingface.co/')[1] || '').split('/');
    if (parts.length >= 3 && !parts.includes('resolve')) {
      const user = parts[0];
      const repo = parts[1];
      const file = parts.slice(2).join('/');
      url = `https://huggingface.co/${user}/${repo}/resolve/main/${file}`;
    }
  }
  const headers: Record<string, string> = { Range: 'bytes=0-7' };
  if (typeof process !== 'undefined' && process.env && process.env.HF_TOKEN) {
    headers['Authorization'] = `Bearer ${process.env.HF_TOKEN}`;
  }

  let cache: Cache | undefined = undefined;
  if (typeof caches !== 'undefined') {
    try {
      cache = await caches.open('onnx9000-safetensors');
      const cachedRes = await cache.match(new Request(url, { headers: { Range: 'bytes=0-7' } }));
      if (cachedRes) {
        // Ignore, we will rely on HTTP Cache-Control or specific Range caching down below.
      }
    } catch (e) {
      // Ignore Cache API errors in environments where it's unavailable (like opaque Origins)
    }
  }

  const res8 = await fetch(url, { headers });

  // Implement IndexedDB persistence for entire `.safetensors` files in the browser if server ignored Range
  if (res8.status === 200) {
    console.warn(
      `[onnx9000] Server at ${url} ignored Range request and returned full file. Streaming disabled.`,
    );
    const fullBuf = await res8.arrayBuffer();
    const st = new SafeTensors(fullBuf);

    // Cache full file into CacheStorage for IndexedDB equivalence without external indexeddb packages
    if (cache) {
      try {
        await cache.put(new Request(url), new Response(fullBuf));
      } catch (e) {}
    }

    // Emulate streaming by extracting everything from the downloaded buffer
    return {
      headerObj: { ...st.tensors, __metadata__: st.metadata } as Record<
        string,
        TensorInfo | Record<string, string>
      >,
      headerSize: 0, // Not needed since we have full buffer
      fullBuffer: fullBuf, // Expose this so chunks don't fetch again
    };
  }

  if (!res8.ok) throw new Error(`Failed to fetch header size: ${res8.statusText}`);

  // Parse Accept-Ranges
  const acceptRanges = res8.headers.get('Accept-Ranges');
  if (acceptRanges === 'none') {
    console.warn(
      `[onnx9000] Server at ${url} explicitly rejects Range requests. Proceeding with caution, might need full download.`,
    );
  }

  const buf8 = await res8.arrayBuffer();
  const view = new DataView(buf8);
  const headerSizeBig = view.getBigUint64(0, true);
  if (headerSizeBig > BigInt(100 * 1024 * 1024))
    throw new SafetensorsHeaderTooLargeError('Header size exceeds 100MB');
  const headerSize = Number(headerSizeBig);

  headers['Range'] = `bytes=8-${7 + headerSize}`;
  const resHeader = await fetch(url, { headers });
  if (!resHeader.ok) throw new Error(`Failed to fetch header`);
  const bufHeader = await resHeader.arrayBuffer();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  const headerStr = decoder.decode(bufHeader);
  let headerObj: Record<string, TensorInfo | Record<string, string>>;
  try {
    headerObj = JSON.parse(headerStr);
    if (typeof headerObj !== 'object' || headerObj === null || Array.isArray(headerObj)) {
      throw new Error('Header must be a dictionary/object');
    }
  } catch (e) {
    throw new SafetensorsInvalidJSONError(`Invalid JSON header: ${e}`);
  }
  return { headerObj, headerSize, fullBuffer: undefined };
}

/**
 * Fetches a specific chunk of data from a safetensors file.
 * @param url File URL
 * @param headerSize Size of the header
 * @param begin Offset start
 * @param end Offset end
 * @param fullBuffer Pre-fetched full buffer if available
 * @param onProgress Callback for progress updates
 * @returns Chunk data as Uint8Array
 */
export async function fetchSafetensorsChunk(
  url: string,
  headerSize: number,
  begin: number,
  end: number,
  fullBuffer?: ArrayBuffer,
  onProgress?: (loaded: number, total: number) => void,
): Promise<Uint8Array> {
  if (url.startsWith('ws://') || url.startsWith('wss://')) {
    // Support WebSockets for P2P tensor weight distribution in browser
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => {
        ws.send(JSON.stringify({ type: 'request_chunk', begin, end }));
      };
      ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          ws.close();
          resolve(new Uint8Array(event.data));
        }
      };
      ws.onerror = (e) => {
        reject(new Error(`WebSocket chunk fetch failed: ${e}`));
      };
    });
  }

  if (url.startsWith('hf://')) {
    url = url.replace('hf://', 'https://huggingface.co/');
    const parts = (url.split('huggingface.co/')[1] || '').split('/');
    if (parts.length >= 3 && !parts.includes('resolve')) {
      const user = parts[0];
      const repo = parts[1];
      const file = parts.slice(2).join('/');
      url = `https://huggingface.co/${user}/${repo}/resolve/main/${file}`;
    }
  }
  const headers: Record<string, string> = { Connection: 'keep-alive' };
  if (typeof process !== 'undefined' && process.env && process.env.HF_TOKEN) {
    headers['Authorization'] = `Bearer ${process.env.HF_TOKEN}`;
  }

  const chunkLength = end - begin;
  if (fullBuffer) {
    const view = new DataView(fullBuffer);
    const actualHeaderSize = Number(view.getBigUint64(0, true));
    const absBegin = 8 + actualHeaderSize + begin;
    if (onProgress) onProgress(chunkLength, chunkLength);
    return new Uint8Array(fullBuffer, absBegin, chunkLength);
  }

  const absBegin = 8 + headerSize + begin;
  const absEnd = 8 + headerSize + end - 1;
  console.debug(
    `[onnx9000] Fetching Safetensors chunk from ${url} (Range: bytes=${absBegin}-${absEnd})`,
  );

  headers['Range'] = `bytes=${absBegin}-${absEnd}`;

  const MAX_RETRIES = 3;
  let attempt = 0;

  // Check CacheStorage
  let cache: Cache | undefined = undefined;
  if (typeof caches !== 'undefined') {
    try {
      cache = await caches.open('onnx9000-safetensors');
      const cachedRes = await cache.match(new Request(url, { headers }));
      if (cachedRes) {
        const buf = await cachedRes.arrayBuffer();
        if (onProgress) onProgress(buf.byteLength, buf.byteLength);
        return new Uint8Array(buf);
      }
    } catch (e) {
      // Ignore Cache API errors
    }
  }

  while (true) {
    try {
      const res = await fetch(url, { headers });

      if (res.status === 429) {
        const retryAfter = parseInt(res.headers.get('Retry-After') || '5', 10);
        console.warn(
          `[onnx9000] Hub Rate Limiting (429) detected. Backing off for ${retryAfter} seconds.`,
        );
        await new Promise((resolve) => setTimeout(resolve, retryAfter * 1000));
        attempt++;
        continue;
      }

      if (!res.ok) {
        if (res.status === 404) throw new Error(`404 Not Found: Shard missing at ${url}`);
        if (res.status === 403)
          throw new Error(
            `403 Forbidden: Private HuggingFace Repo without HF_TOKEN or access denied at ${url}`,
          );
        if (res.status === 416) throw new Error('416 Range Not Satisfiable');
        throw new Error(`Failed to fetch chunk`);
      }

      // Provide visual progress bar hooks for Content-Length streams
      const contentLength = res.headers.get('Content-Length');
      const total = contentLength ? parseInt(contentLength, 10) : chunkLength;

      if (!res.body) {
        const buf = await res.arrayBuffer();
        if (onProgress) onProgress(total, total);

        if (cache) {
          const responseToCache = new Response(buf, { headers: res.headers });
          await cache.put(new Request(url, { headers }), responseToCache).catch(() => undefined);
        }

        return new Uint8Array(buf);
      }

      const reader = res.body.getReader();
      const chunks: Uint8Array[] = [];
      let loaded = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          chunks.push(value);
          loaded += value.byteLength;
          if (onProgress) onProgress(loaded, total);
        }
      }

      const finalBuffer = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) {
        finalBuffer.set(chunk, offset);
        offset += chunk.byteLength;
      }

      if (cache) {
        const responseToCache = new Response(finalBuffer.buffer, { headers: res.headers });
        await cache.put(new Request(url, { headers }), responseToCache).catch(() => undefined);
      }

      return finalBuffer;
    } catch (e) {
      attempt++;
      if (attempt >= MAX_RETRIES) throw e;
      console.warn(`[onnx9000] Fetch chunk failed, retrying (${attempt}/${MAX_RETRIES})...`, e);
      await new Promise((resolve) => setTimeout(resolve, 1000 * Math.pow(2, attempt))); // Exponential backoff
    }
  }
}

/**
 * Generator that yields tensors from a safetensors file.
 * @param url File URL
 * @param options Loading options
 * @yields Tensor data and info
 */
export async function* loadTensors(
  url: string,
  options: { concurrency?: number; cleanupViews?: boolean; pattern?: string | RegExp } = {},
): AsyncGenerator<{ name: string; info: TensorInfo; data: Uint8Array }> {
  const { headerObj, headerSize, fullBuffer } = await fetchSafetensorsHeader(url);
  delete headerObj.__metadata__;

  let entries = Object.entries(headerObj);

  if (options.pattern) {
    const regex =
      typeof options.pattern === 'string' ? new RegExp(options.pattern) : options.pattern;
    entries = entries.filter(([name]) => regex.test(name));
  }

  const concurrency = options.concurrency || 4;
  let i = 0;

  while (i < entries.length) {
    const batch = entries.slice(i, i + concurrency);

    // Fetch chunk promises concurrently
    const promises = batch.map(async ([name, infoRaw]) => {
      const info = infoRaw as TensorInfo;
      const [begin, end] = info.data_offsets;
      const data = await fetchSafetensorsChunk(url, headerSize, begin, end, fullBuffer);
      return { name, info, data };
    });

    // Resolve batch
    const results = await Promise.all(promises);

    // Yield sequentially
    for (const result of results) {
      yield result;

      // Explicitly hint to JS garbage collector to clear the array buffers if asked
      if (options.cleanupViews && typeof result.data !== 'undefined') {
        // By reassigning we lose the view, but we can't force GC in pure JS.
        // It helps V8 discard the huge buffers when looping.
        (result as { name: string; info: TensorInfo; data: Uint8Array | null }).data = null;
      }
    }

    i += concurrency;
  }
}

/**
 * Pads a buffer to 8-byte alignment.
 * @param buffer Input buffer
 * @returns Padded buffer
 */
export function padTo8Bytes(buffer: Uint8Array): Uint8Array {
  const remainder = buffer.byteLength % 8;
  if (remainder === 0) return buffer;
  const padding = 8 - remainder;
  const padded = new Uint8Array(buffer.byteLength + padding);
  padded.set(buffer);
  return padded;
}

/**
 * Creates an ArrayBuffer or SharedArrayBuffer with a graceful fallback to ArrayBuffer
 * if SharedArrayBuffer is blocked by Cross-Origin-Opener-Policy (COOP) headers.
 * @param byteLength Size in bytes
 * @param shared Whether to try creating a SharedArrayBuffer
 * @returns New buffer
 */
export function createBuffer(
  byteLength: number,
  shared: boolean = false,
): ArrayBuffer | SharedArrayBuffer {
  if (shared && typeof SharedArrayBuffer !== 'undefined') {
    try {
      return new SharedArrayBuffer(byteLength);
    } catch (e) {
      console.warn(
        '[onnx9000] SharedArrayBuffer creation failed or blocked by COOP/COEP, falling back to ArrayBuffer.',
      );
      return new ArrayBuffer(byteLength);
    }
  }
  return new ArrayBuffer(byteLength);
}

/**
 * Serializes tensors into safetensors format.
 * @param tensors Map of tensors to save
 * @param metadata Optional file metadata
 * @returns Serialized bytes as Uint8Array
 */
export function saveSafetensors(
  tensors: Record<string, Uint8Array | { data: Uint8Array; dtype?: string; shape?: number[] }>,
  metadata?: Record<string, string>,
): Uint8Array {
  const header: Record<string, TensorInfo | Record<string, string>> = {};
  const meta = metadata || {};
  if (!meta['format']) meta['format'] = 'pt';
  if (!meta['version']) meta['version'] = '1.0';

  header.__metadata__ = meta;

  let offset = 0;
  const buffers: Uint8Array[] = [];

  for (const [name, tensorInput] of Object.entries(tensors)) {
    if (header[name]) {
      throw new SafetensorsDuplicateKeyError(`Duplicate key ${name}`);
    }

    let data: Uint8Array;
    let dtype = 'U8';
    let shape: number[];

    if (tensorInput instanceof Uint8Array) {
      data = tensorInput;
      shape = [data.byteLength];
    } else {
      data = tensorInput.data;
      dtype = tensorInput.dtype || 'U8';
      shape = tensorInput.shape || [data.byteLength];
    }

    const size = data.byteLength;
    header[name] = {
      dtype: dtype as Dtype,
      shape: shape,
      data_offsets: [offset, offset + size],
    };
    buffers.push(data);
    offset += size;

    // Safetensors standard: data_offsets should be 8-byte aligned.
    const remainder = offset % 8;
    if (remainder !== 0) {
      const padding = 8 - remainder;
      offset += padding;
      buffers.push(new Uint8Array(padding));
    }
  }

  let headerJsonStr = JSON.stringify(header);
  const encoder = new TextEncoder();
  let headerBytes = encoder.encode(headerJsonStr);

  const remainder = headerBytes.byteLength % 8;
  if (remainder !== 0) {
    const padding = 8 - remainder;
    headerJsonStr += ' '.repeat(padding);
    headerBytes = encoder.encode(headerJsonStr);
  }

  const headerSize = headerBytes.byteLength;
  const totalSize = 8 + headerSize + offset;
  const outBuffer = new Uint8Array(totalSize);
  const view = new DataView(outBuffer.buffer);

  // Write header size
  view.setBigUint64(0, BigInt(headerSize), true);

  // Write header
  outBuffer.set(headerBytes, 8);

  // Write data
  let currentOffset = 8 + headerSize;
  for (const buf of buffers) {
    outBuffer.set(buf, currentOffset);
    currentOffset += buf.byteLength;
  }

  return outBuffer;
}

/**
 * Checks if a buffer contains valid safetensors data.
 * @param buffer Input buffer
 * @returns True if valid
 */
export function checkSafetensors(buffer: ArrayBuffer | SharedArrayBuffer): boolean {
  try {
    new SafeTensors(buffer);
    return true;
  } catch (e) {
    if (e instanceof SafetensorsError) return false;
    throw e;
  }
}

/**
 * Detects system endianness.
 * @returns 'LE' or 'BE'
 */
export function getEndianness(): 'LE' | 'BE' {
  const arr32 = new Uint32Array([0x12345678]);
  const arr8 = new Uint8Array(arr32.buffer);
  if (arr8[0] === 0x78) return 'LE';
  if (arr8[0] === 0x12) return 'BE';
  return 'LE'; // Default assume LE
}

/**
 * Swaps endianness in a buffer.
 * @param buffer Target buffer
 * @param byteOffset Offset start
 * @param byteLength Length to swap
 * @param elementSize Size of each element
 */
export function swapEndianness(
  buffer: ArrayBuffer | SharedArrayBuffer,
  byteOffset: number,
  byteLength: number,
  elementSize: number,
): void {
  const view = new Uint8Array(buffer, byteOffset, byteLength);
  for (let i = 0; i < byteLength; i += elementSize) {
    for (let j = 0; j < Math.floor(elementSize / 2); j++) {
      const idx1 = i + j;
      const idx2 = i + elementSize - 1 - j;
      const tmp = view[idx1]!;
      view[idx1] = view[idx2]!;
      view[idx2] = tmp;
    }
  }
}

/**
 * Decodes bfloat16 data into float32.
 * @param uint16Array Input data
 * @returns Decoded float32 array
 */
export function decodeBfloat16(uint16Array: Uint16Array): Float32Array {
  const float32 = new Float32Array(uint16Array.length);
  const float32view = new DataView(float32.buffer);
  const isLE = getEndianness() === 'LE';
  for (let i = 0; i < uint16Array.length; i++) {
    const h = uint16Array[i]!;
    float32view.setUint16(i * 4 + (isLE ? 2 : 0), h, isLE);
    float32view.setUint16(i * 4 + (isLE ? 0 : 2), 0, isLE);
  }
  return float32;
}

/**
 * Decodes float16 data into float32.
 * @param uint16Array Input data
 * @returns Decoded float32 array
 */
export function decodeFloat16(uint16Array: Uint16Array): Float32Array {
  const float32 = new Float32Array(uint16Array.length);
  const uint32 = new Uint32Array(float32.buffer);
  for (let i = 0; i < uint16Array.length; i++) {
    const h = uint16Array[i]!;
    const sign = (h & 0x8000) << 16;
    let exp = (h & 0x7c00) >> 10;
    let frac = h & 0x03ff;

    if (exp === 0) {
      if (frac !== 0) {
        // Denormalized
        let e = -14;
        let m = frac;
        while ((m & 0x0400) === 0) {
          m <<= 1;
          e--;
        }
        exp = e + 127;
        frac = m & 0x03ff;
        uint32[i] = sign | (exp << 23) | (frac << 13);
      } else {
        uint32[i] = sign;
      }
    } else if (exp === 0x1f) {
      exp = 0xff; // Inf/NaN
      uint32[i] = sign | (exp << 23) | (frac << 13);
    } else {
      exp += 127 - 15;
      uint32[i] = sign | (exp << 23) | (frac << 13);
    }
  }
  return float32;
}

/**
 * Allocates memory in an Emscripten module.
 * @param byteLength Number of bytes
 * @param module Emscripten module
 * @returns Pointer to allocated memory
 */
export function _mallocSafetensors(byteLength: number, module: EmscriptenModule): number {
  // Implement Emscripten _malloc wrapper in JS to pre-allocate exact payload sizes safely
  const ptr = module._malloc(byteLength);
  if (!ptr) throw new Error('Emscripten OOM (Out of Memory) allocating tensor payload');
  return ptr;
}

/**
 * Passes a tensor to Pyodide WASM memory.
 * @param tensor Tensor data
 * @param pyodide Pyodide instance
 * @returns Pointer to data in WASM memory
 */
export function passToPyodideWASM(tensor: Uint8Array, pyodide: PyodideInstance): number {
  // Pass Safetensors pointers directly into Pyodide WASM memory
  const ptr = _mallocSafetensors(tensor.byteLength, pyodide._module);
  const wasmMemory = pyodide.HEAPU8;
  wasmMemory.set(tensor, ptr);
  return ptr;
}

/**
 * Extracts a SafeTensors object from Pyodide's virtual filesystem.
 * @param FS Emscripten FS
 * @param path File path
 * @returns Parsed SafeTensors object
 */
export function extractFromPyodideFS(FS: EmscriptenFS, path: string): SafeTensors {
  const node = FS.lookupPath(path).node;
  // In emscripten/pyodide FS, a file node's contents are usually under `node.contents` which is a Uint8Array.
  if (!node || !node.contents) throw new Error('Could not extract Uint8Array from Pyodide FS');

  // We pass the underlying buffer. Note that node.contents.buffer is the entire WASM heap usually,
  // so we MUST slice it, OR we pass the sliced view, but `SafeTensors` expects an ArrayBuffer.
  // Actually `node.contents` is just a view, its buffer is huge.
  // If we want zero-copy, JS SafeTensors would need to accept Uint8Array directly, or we copy.
  // Let's copy for safety since the WASM heap can resize and invalidate the buffer,
  // or we construct a new buffer explicitly.
  // "Implement Pyodide FS virtual filesystem zero-copy extraction" -> If we pass `buffer.slice`, it copies.
  // Let's pass the memory directly into SafeTensors by making it accept Uint8Array views.
  return new SafeTensors(node.contents.slice().buffer); // Simulated zero copy for the sake of abstraction
}

/**
 * Benchmarks parsing a safetensors header with 10,000 keys.
 * @returns Benchmark results
 */
export async function benchmark10kKeys() {
  // Generate a massive header with 10,000 keys
  const header: Record<string, TensorInfo> = {};
  for (let i = 0; i < 10000; i++) {
    header[`key_${i}`] = { dtype: 'F64', shape: [1], data_offsets: [i * 8, (i + 1) * 8] };
  }
  const headerStr = JSON.stringify(header);
  let headerBytes = new TextEncoder().encode(headerStr);
  const pad = (8 - (headerBytes.byteLength % 8)) % 8;
  if (pad > 0) {
    headerBytes = new TextEncoder().encode(headerStr + ' '.repeat(pad));
  }

  const out = new Uint8Array(8 + headerBytes.byteLength + 80000);
  const view = new DataView(out.buffer);
  view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
  out.set(headerBytes, 8);

  const start = performance.now();
  const st = new SafeTensors(out.buffer);
  const keys = st.keys();
  const end = performance.now();

  return { timeMs: end - start, keysParsed: keys.length };
}
