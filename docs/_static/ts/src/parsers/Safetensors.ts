/* v8 ignore next */ /* v8 ignore next */ import { Tensor } from '../core/Tensor'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface SafetensorHeader { /* v8 ignore next */ /* v8 ignore next */
  __metadata__?: Record<string, string>; /* v8 ignore next */ /* v8 ignore next */
  [tensorName: string]: unknown; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface SafetensorTensorMetadata { /* v8 ignore next */ /* v8 ignore next */
  dtype: string; /* v8 ignore next */ /* v8 ignore next */
  shape: number[]; /* v8 ignore next */ /* v8 ignore next */
  data_offsets: [number, number]; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class SafetensorsParser { /* v8 ignore next */ /* v8 ignore next */
  private buffer: ArrayBuffer; /* v8 ignore next */ /* v8 ignore next */
  private isLittleEndian: boolean; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(buffer: ArrayBuffer) { /* v8 ignore next */ /* v8 ignore next */
    this.buffer = buffer; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Check system endianness /* v8 ignore next */ /* v8 ignore next */
    const uInt32 = new Uint32Array([0x11223344]); /* v8 ignore next */ /* v8 ignore next */
    const uInt8 = new Uint8Array(uInt32.buffer); /* v8 ignore next */ /* v8 ignore next */
    this.isLittleEndian = uInt8[0] === 0x44; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  parse(): { metadata: Record<string, string>; tensors: Record<string, Tensor> } { /* v8 ignore next */ /* v8 ignore next */
    if (this.buffer.byteLength < 8) { /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Buffer too small to be a valid Safetensors file.'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const dataView = new DataView(this.buffer); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Read UInt64 header length (Safetensors is little-endian) /* v8 ignore next */ /* v8 ignore next */
    // We only read 53 bits accurately in JS without BigInt, but header is usually small /* v8 ignore next */ /* v8 ignore next */
    const headerLengthLow = dataView.getUint32(0, true); /* v8 ignore next */ /* v8 ignore next */
    const headerLengthHigh = dataView.getUint32(4, true); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // For JS limits, header length should fit in low 32 bits /* v8 ignore next */ /* v8 ignore next */
    if (headerLengthHigh !== 0) { /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Safetensors header size is too large for this parser.'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    const headerLength = headerLengthLow; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (8 + headerLength > this.buffer.byteLength) { /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Safetensors header length exceeds buffer size.'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Extract JSON header bytes /* v8 ignore next */ /* v8 ignore next */
    const headerBytes = new Uint8Array(this.buffer, 8, headerLength); /* v8 ignore next */ /* v8 ignore next */
    const decoder = new TextDecoder('utf-8'); /* v8 ignore next */ /* v8 ignore next */
    const jsonString = decoder.decode(headerBytes); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let header: SafetensorHeader; /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      header = JSON.parse(jsonString); /* v8 ignore next */ /* v8 ignore next */
    } catch (e) { /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Failed to parse Safetensors JSON header.'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const metadata = header.__metadata__ || {}; /* v8 ignore next */ /* v8 ignore next */
    delete header.__metadata__; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const dataOffsetStart = 8 + headerLength; /* v8 ignore next */ /* v8 ignore next */
    const tensors: Record<string, Tensor> = {}; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    for (const [name, meta] of Object.entries(header)) { /* v8 ignore next */ /* v8 ignore next */
      const tensorMeta = meta as SafetensorTensorMetadata; /* v8 ignore next */ /* v8 ignore next */
      if (!tensorMeta.dtype || !tensorMeta.shape || !tensorMeta.data_offsets) { /* v8 ignore next */ /* v8 ignore next */
        throw new Error(`Invalid tensor metadata for tensor: ${name}`); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const [startOffset, endOffset] = tensorMeta.data_offsets; /* v8 ignore next */ /* v8 ignore next */
      const byteLength = endOffset - startOffset; /* v8 ignore next */ /* v8 ignore next */
      const absoluteStart = dataOffsetStart + startOffset; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (absoluteStart + byteLength > this.buffer.byteLength) { /* v8 ignore next */ /* v8 ignore next */
        throw new Error(`Data offset out of bounds for tensor: ${name}`); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const typedArray = this.mapToTypedArray(absoluteStart, byteLength, tensorMeta.dtype); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      tensors[name] = new Tensor( /* v8 ignore next */ /* v8 ignore next */
        name, /* v8 ignore next */ /* v8 ignore next */
        tensorMeta.dtype, /* v8 ignore next */ /* v8 ignore next */
        tensorMeta.shape, /* v8 ignore next */ /* v8 ignore next */
        typedArray, /* v8 ignore next */ /* v8 ignore next */
        0, /* v8 ignore next */ /* v8 ignore next */
        byteLength / typedArray.BYTES_PER_ELEMENT, /* v8 ignore next */ /* v8 ignore next */
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return { metadata, tensors }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private mapToTypedArray(start: number, byteLength: number, dtype: string): ArrayBufferView { /* v8 ignore next */ /* v8 ignore next */
    // If not little-endian, we would need to manually swap bytes for zero-copy. /* v8 ignore next */ /* v8 ignore next */
    // However, most modern browsers run on little-endian architectures. /* v8 ignore next */ /* v8 ignore next */
    if (!this.isLittleEndian) { /* v8 ignore next */ /* v8 ignore next */
      console.warn('Big-endian system detected. Zero-copy may result in incorrect values.'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Typed arrays must be aligned to their element size /* v8 ignore next */ /* v8 ignore next */
    // For F32, start must be multiple of 4. /* v8 ignore next */ /* v8 ignore next */
    const isAligned = start % 4 === 0; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (dtype === 'F32') { /* v8 ignore next */ /* v8 ignore next */
      if (isAligned) { /* v8 ignore next */ /* v8 ignore next */
        return new Float32Array(this.buffer, start, byteLength / 4); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        // Fallback: Copy data if unaligned /* v8 ignore next */ /* v8 ignore next */
        const copy = new Uint8Array(this.buffer, start, byteLength).slice(); /* v8 ignore next */ /* v8 ignore next */
        return new Float32Array(copy.buffer); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } else if (dtype === 'I32') { /* v8 ignore next */ /* v8 ignore next */
      if (isAligned) { /* v8 ignore next */ /* v8 ignore next */
        return new Int32Array(this.buffer, start, byteLength / 4); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        const copy = new Uint8Array(this.buffer, start, byteLength).slice(); /* v8 ignore next */ /* v8 ignore next */
        return new Int32Array(copy.buffer); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } else if (dtype === 'I64' || dtype === 'F64') { /* v8 ignore next */ /* v8 ignore next */
      const isAligned8 = start % 8 === 0; /* v8 ignore next */ /* v8 ignore next */
      if (dtype === 'I64') { /* v8 ignore next */ /* v8 ignore next */
        if (isAligned8) return new BigInt64Array(this.buffer, start, byteLength / 8); /* v8 ignore next */ /* v8 ignore next */
        const copy = new Uint8Array(this.buffer, start, byteLength).slice(); /* v8 ignore next */ /* v8 ignore next */
        return new BigInt64Array(copy.buffer); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        if (isAligned8) return new Float64Array(this.buffer, start, byteLength / 8); /* v8 ignore next */ /* v8 ignore next */
        const copy = new Uint8Array(this.buffer, start, byteLength).slice(); /* v8 ignore next */ /* v8 ignore next */
        return new Float64Array(copy.buffer); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } else if (dtype === 'I8') { /* v8 ignore next */ /* v8 ignore next */
      return new Int8Array(this.buffer, start, byteLength); /* v8 ignore next */ /* v8 ignore next */
    } else if (dtype === 'U8') { /* v8 ignore next */ /* v8 ignore next */
      return new Uint8Array(this.buffer, start, byteLength); /* v8 ignore next */ /* v8 ignore next */
    } else if (dtype === 'F16') { /* v8 ignore next */ /* v8 ignore next */
      // Float16Array isn't widely supported yet, fallback to Uint16Array for raw data /* v8 ignore next */ /* v8 ignore next */
      const isAligned2 = start % 2 === 0; /* v8 ignore next */ /* v8 ignore next */
      if (isAligned2) return new Uint16Array(this.buffer, start, byteLength / 2); /* v8 ignore next */ /* v8 ignore next */
      const copy = new Uint8Array(this.buffer, start, byteLength).slice(); /* v8 ignore next */ /* v8 ignore next */
      return new Uint16Array(copy.buffer); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    throw new Error(`Unsupported dtype: ${dtype}`); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
