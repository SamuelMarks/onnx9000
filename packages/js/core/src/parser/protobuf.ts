export const WIRE_TYPE_VARINT = 0;
export const WIRE_TYPE_64BIT = 1;
export const WIRE_TYPE_LENGTH_DELIMITED = 2;
export const WIRE_TYPE_START_GROUP = 3;
export const WIRE_TYPE_END_GROUP = 4;
export const WIRE_TYPE_32BIT = 5;

export interface Reader {
  readByte(): Promise<number>;
  readBytes(length: number): Promise<Uint8Array>;
  skip(length: number): Promise<void>;
  getPosition(): number;
  getLength(): number;
}

export class BufferReader implements Reader {
  private buffer: Uint8Array;
  private offset: number = 0;

  constructor(buffer: Uint8Array) {
    this.buffer = buffer;
  }

  async readByte(): Promise<number> {
    if (this.offset >= this.buffer.length) {
      throw new Error('Unexpected end of buffer');
    }
    return this.buffer[this.offset++]!;
  }

  async readBytes(length: number): Promise<Uint8Array> {
    if (this.offset + length > this.buffer.length) {
      throw new Error('Unexpected end of buffer');
    }
    const res = this.buffer.subarray(this.offset, this.offset + length);
    this.offset += length;
    return res;
  }

  async skip(length: number): Promise<void> {
    this.offset += length;
  }

  getPosition(): number {
    return this.offset;
  }

  getLength(): number {
    return this.buffer.length;
  }
}

export class BlobReader implements Reader {
  private blob: Blob;
  private offset: number = 0;
  private cache: Uint8Array | null = null;
  private cacheStart: number = 0;
  private cacheSize: number = 1024 * 1024 * 4; // 4MB cache

  constructor(blob: Blob, cacheSize?: number) {
    this.blob = blob;
    if (cacheSize) {
      this.cacheSize = cacheSize;
    }
  }

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

  async readByte(): Promise<number> {
    if (this.offset >= this.blob.size) {
      throw new Error('Unexpected end of blob');
    }
    await this.ensureCache(1);
    const byte = this.cache![this.offset - this.cacheStart]!;
    this.offset++;
    return byte;
  }

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

  async skip(length: number): Promise<void> {
    this.offset += length;
  }

  getPosition(): number {
    return this.offset;
  }

  getLength(): number {
    return this.blob.size;
  }
}

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

export async function readString(reader: Reader, length: number): Promise<string> {
  const bytes = await reader.readBytes(length);
  return new TextDecoder().decode(bytes);
}

export async function readTag(reader: Reader): Promise<{ fieldNumber: number; wireType: number }> {
  const tag = await readVarInt(reader);
  return { fieldNumber: tag >> 3, wireType: tag & 7 };
}

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
