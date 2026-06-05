/** Protobuf Varint wire type. */
export const WIRE_TYPE_VARINT = 0;
/** Protobuf 64-bit wire type. */
export const WIRE_TYPE_64BIT = 1;
/** Protobuf Length-delimited wire type. */
export const WIRE_TYPE_LENGTH_DELIMITED = 2;
/** Protobuf Start Group wire type (deprecated). */
export const WIRE_TYPE_START_GROUP = 3;
/** Protobuf End Group wire type (deprecated). */
export const WIRE_TYPE_END_GROUP = 4;
/** Protobuf 32-bit wire type. */
export const WIRE_TYPE_32BIT = 5;

/**
 * Interface for reading protobuf data from various sources.
 */
export interface Reader {
  /** Read a single byte. */
  readByte(): Promise<number>;
  /** Read a fixed number of bytes. */
  readBytes(length: number): Promise<Uint8Array>;
  /** Skip a fixed number of bytes. */
  skip(length: number): Promise<void>;
  /** Get current byte position. */
  getPosition(): number;
  /** Get total length of data. */
  getLength(): number;
}

/**
 * Reader for Uint8Array buffers.
 */
export class BufferReader implements Reader {
  private buffer: Uint8Array;
  private offset: number = 0;

  /**
   * Create a new BufferReader.
   * @param buffer The input buffer.
   */
  constructor(buffer: Uint8Array) {
    this.buffer = buffer;
  }

  /** Read a single byte from the buffer. */
  async readByte(): Promise<number> {
    if (this.offset >= this.buffer.length) {
      throw new Error('Unexpected end of buffer');
    }
    return this.buffer[this.offset++]!;
  }

  /** Read a fixed number of bytes from the buffer. */
  async readBytes(length: number): Promise<Uint8Array> {
    if (this.offset + length > this.buffer.length) {
      throw new Error('Unexpected end of buffer');
    }
    const res = this.buffer.subarray(this.offset, this.offset + length);
    this.offset += length;
    return res;
  }

  /** Skip bytes in the buffer. */
  async skip(length: number): Promise<void> {
    this.offset += length;
  }

  /** Get current offset. */
  getPosition(): number {
    return this.offset;
  }

  /** Get buffer length. */
  getLength(): number {
    return this.buffer.length;
  }
}

/**
 * Reader for Blob objects, using a rolling cache for efficiency.
 */
export class BlobReader implements Reader {
  private blob: Blob;
  private offset: number = 0;
  private cache: Uint8Array | null = null;
  private cacheStart: number = 0;
  private cacheSize: number = 1024 * 1024 * 4; // 4MB cache

  /**
   * Create a new BlobReader.
   * @param blob The input Blob.
   * @param cacheSize Optional cache size in bytes.
   */
  constructor(blob: Blob, cacheSize?: number) {
    this.blob = blob;
    if (cacheSize) {
      this.cacheSize = cacheSize;
    }
  }

  /** Ensure enough data is in the local cache. */
  private async ensureCache(length: number): Promise<void> {
    if (
      !this.cache ||
      this.offset < this.cacheStart ||
      this.offset + length > this.cacheStart + this.cache.length
    ) {
      const start = this.offset;
      const end = Math.min(this.blob.size, start + Math.max(this.cacheSize, length));
      const slice = this.blob.slice(start, end);
      const arrayBuffer = await slice.arrayBuffer();
      this.cache = new Uint8Array(arrayBuffer);
      this.cacheStart = start;
    }
  }

  /** Read a single byte from the blob. */
  async readByte(): Promise<number> {
    if (this.offset >= this.blob.size) {
      throw new Error('Unexpected end of blob');
    }
    await this.ensureCache(1);
    const byte = this.cache![this.offset - this.cacheStart]!;
    this.offset++;
    return byte;
  }

  /** Read a fixed number of bytes from the blob. */
  async readBytes(length: number): Promise<Uint8Array> {
    if (this.offset + length > this.blob.size) {
      throw new Error('Unexpected end of blob');
    }
    await this.ensureCache(length);
    const start = this.offset - this.cacheStart;
    const res = this.cache!.subarray(start, start + length);
    this.offset += length;
    return res;
  }

  /** Skip bytes in the blob. */
  async skip(length: number): Promise<void> {
    this.offset += length;
  }

  /** Get current offset. */
  getPosition(): number {
    return this.offset;
  }

  /** Get blob size. */
  getLength(): number {
    return this.blob.size;
  }
}

/**
 * Read a 32-bit VarInt from the reader.
 * @param reader Input reader.
 */
export async function readVarInt(reader: Reader): Promise<number> {
  let result = 0;
  let shift = 0;
  while (true) {
    const byte = await reader.readByte();
    result |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) {
      return result;
    }
    shift += 7;
  }
}

/**
 * Read a 64-bit VarInt from the reader as a BigInt.
 * @param reader Input reader.
 */
export async function readVarInt64(reader: Reader): Promise<bigint> {
  let result = 0n;
  let shift = 0n;
  while (true) {
    const byte = await reader.readByte();
    result |= BigInt(byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) {
      return result;
    }
    shift += 7n;
  }
}

/**
 * Read a string from the reader.
 * @param reader Input reader.
 * @param length Length of the string in bytes.
 */
export async function readString(reader: Reader, length: number): Promise<string> {
  const bytes = await reader.readBytes(length);
  return new TextDecoder().decode(bytes);
}

/**
 * Read a field tag (field number and wire type).
 * @param reader Input reader.
 */
export async function readTag(reader: Reader): Promise<{ fieldNumber: number; wireType: number }> {
  const tag = await readVarInt(reader);
  return { fieldNumber: tag >> 3, wireType: tag & 7 };
}

/**
 * Skip a field based on its wire type.
 * @param reader Input reader.
 * @param wireType Protobuf wire type.
 */
export async function skipField(reader: Reader, wireType: number): Promise<void> {
  switch (wireType) {
    case WIRE_TYPE_VARINT:
      await readVarInt64(reader);
      break;
    case WIRE_TYPE_64BIT:
      await reader.skip(8);
      break;
    case WIRE_TYPE_LENGTH_DELIMITED:
      const length = await readVarInt(reader);
      await reader.skip(length);
      break;
    case WIRE_TYPE_START_GROUP:
    case WIRE_TYPE_END_GROUP:
      throw new Error('Groups are not supported');
    case WIRE_TYPE_32BIT:
      await reader.skip(4);
      break;
    default:
      throw new Error(`Unsupported wire type: ${wireType}`);
  }
}
