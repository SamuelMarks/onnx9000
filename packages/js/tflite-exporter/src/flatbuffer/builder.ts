/**
 * @fileoverview builder module for tflite-exporter.
 */
/**
 * Zero-dependency FlatBuffer Builder in TypeScript.
 */
import { Onnx9000Error } from '@onnx9000/core';
/** Represents the `FlatBufferBuilder` class. */
export class FlatBufferBuilder {
  /** Property `bb` of `FlatBufferBuilder`. */
  private bb: Uint8Array;
  /** Property `space` of `FlatBufferBuilder`. */
  private space: number;
  /** Property `minalign` of `FlatBufferBuilder`. */
  private minalign: number = 1;
  /** Property `vtable` of `FlatBufferBuilder`. */
  private vtable: number[] | null = null;
  /** Property `objectStart` of `FlatBufferBuilder`. */
  private objectStart: number = 0;
  /** Property `vtables` of `FlatBufferBuilder`. */
  private vtables: number[] = [];
  /** Property `view` of `FlatBufferBuilder`. */
  private view: DataView;

  /**
   * Initializes a new `FlatBufferBuilder` instance.
   * @param initialSize Parameter `initialSize`.
   */
  constructor(initialSize: number = 1024) {
    this.bb = new Uint8Array(initialSize);
    this.view = new DataView(this.bb.buffer);
    this.space = initialSize;
  }

  /** Executes `clear` on `FlatBufferBuilder`. */
  public clear(): void {
    this.space = this.bb.length;
    this.minalign = 1;
    this.vtable = null;
    this.objectStart = 0;
    this.vtables = [];
  }

  /** Executes `growBuffer` on `FlatBufferBuilder`. */
  private growBuffer(): void {
    const oldLength = this.bb.length;
    let newLength = oldLength * 2;
    if (newLength === 0) newLength = 1024;

    // 25. Support chunked writing for models exceeding JS ArrayBuffer limits (>2GB).
    // V8 ArrayBuffer max length is ~2GB (2,147,483,647 bytes on 32-bit architecture limits, up to 4GB on 64-bit Chrome).
    // Node.js max is typically ~2GB buffer size.
    const MAX_ARRAY_BUFFER_SIZE = 2147483647; // 2GB limit
    if (newLength > MAX_ARRAY_BUFFER_SIZE) {
      if (oldLength < MAX_ARRAY_BUFFER_SIZE) {
        newLength = MAX_ARRAY_BUFFER_SIZE; // Push exactly to the limit first
      } else {
        throw new Onnx9000Error(
          `[onnx2tf] Fatal: TFLite model exceeds absolute v8 ArrayBuffer maximum limits (2GB). Use chunked streaming writer or target Python native exporter for massive parameter isolation.`,
        );
      }
    }

    const newBb = new Uint8Array(newLength);
    newBb.set(this.bb, newLength - oldLength);
    this.bb = newBb;
    this.view = new DataView(this.bb.buffer);
    this.space += newLength - oldLength;
  }

  /**
   * Executes `prep` on `FlatBufferBuilder`.
   * @param size Parameter `size`.
   * @param additionalBytes Parameter `additionalBytes`.
   */
  private prep(size: number, additionalBytes: number): void {
    if (size > this.minalign) {
      this.minalign = size;
    }
    const alignSize = (~(this.bb.length - this.space + additionalBytes) + 1) & (size - 1);
    while (this.space < alignSize + size + additionalBytes) {
      this.growBuffer();
    }
    this.pad(alignSize);
  }

  /**
   * Executes `pad` on `FlatBufferBuilder`.
   * @param bytes Parameter `bytes`.
   */
  private pad(bytes: number): void {
    for (let i = 0; i < bytes; i++) {
      this.bb[--this.space] = 0;
    }
  }

  /**
   * Executes `place` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public place(x: number): void {
    this.space -= 4;
    this.view.setInt32(this.space, x, true);
  }

  /**
   * Executes `placeInt8` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public placeInt8(x: number): void {
    this.bb[--this.space] = x;
  }

  /**
   * Executes `placeInt16` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public placeInt16(x: number): void {
    this.space -= 2;
    this.view.setInt16(this.space, x, true);
  }

  /**
   * Executes `placeInt32` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public placeInt32(x: number): void {
    this.space -= 4;
    this.view.setInt32(this.space, x, true);
  }

  /**
   * Executes `placeInt64` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public placeInt64(x: bigint): void {
    this.space -= 8;
    this.view.setBigInt64(this.space, x, true);
  }

  /**
   * Executes `placeFloat32` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public placeFloat32(x: number): void {
    this.space -= 4;
    this.view.setFloat32(this.space, x, true);
  }

  /**
   * Executes `placeFloat64` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public placeFloat64(x: number): void {
    this.space -= 8;
    this.view.setFloat64(this.space, x, true);
  }

  /**
   * Executes `addInt8` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public addInt8(x: number): void {
    this.prep(1, 0);
    this.placeInt8(x);
  }

  /**
   * Executes `addInt16` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public addInt16(x: number): void {
    this.prep(2, 0);
    this.placeInt16(x);
  }

  /**
   * Executes `addInt32` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public addInt32(x: number): void {
    this.prep(4, 0);
    this.placeInt32(x);
  }

  /**
   * Executes `addInt64` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public addInt64(x: bigint): void {
    this.prep(8, 0);
    this.placeInt64(x);
  }

  /**
   * Executes `addFloat32` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public addFloat32(x: number): void {
    this.prep(4, 0);
    this.placeFloat32(x);
  }

  /**
   * Executes `addFloat64` on `FlatBufferBuilder`.
   * @param x Parameter `x`.
   */
  public addFloat64(x: number): void {
    this.prep(8, 0);
    this.placeFloat64(x);
  }

  /**
   * Executes `addOffset` on `FlatBufferBuilder`.
   * @param offset Parameter `offset`.
   */
  public addOffset(offset: number): void {
    this.prep(4, 0);
    this.place(this.offset() - offset + 4);
  }

  /**
   * Executes `startVector` on `FlatBufferBuilder`.
   * @param elemSize Parameter `elemSize`.
   * @param numElems Parameter `numElems`.
   * @param alignment Parameter `alignment`.
   */
  public startVector(elemSize: number, numElems: number, alignment: number): void {
    this.prep(4, elemSize * numElems);
    this.prep(alignment, elemSize * numElems);
  }

  /**
   * Executes `endVector` on `FlatBufferBuilder`.
   * @param numElems Parameter `numElems`.
   * @returns Result of `endVector`.
   */
  public endVector(numElems: number): number {
    this.placeInt32(numElems);
    return this.offset();
  }

  /**
   * Executes `createByteVector` on `FlatBufferBuilder`.
   * @param data Parameter `data`.
   * @param alignment Parameter `alignment`.
   * @returns Result of `createByteVector`.
   */
  public createByteVector(data: Uint8Array | number[], alignment: number = 4): number {
    this.startVector(1, data.length, alignment);
    this.space -= data.length;
    this.bb.set(data, this.space);
    return this.endVector(data.length);
  }

  /**
   * Executes `createString` on `FlatBufferBuilder`.
   * @param s Parameter `s`.
   * @returns Result of `createString`.
   */
  public createString(s: string | Uint8Array): number {
    let utf8: Uint8Array;
    if (typeof s === 'string') {
      utf8 = new TextEncoder().encode(s);
    } else {
      utf8 = s;
    }
    this.addInt8(0);
    this.startVector(1, utf8.length, 1);
    this.space -= utf8.length;
    this.bb.set(utf8, this.space);
    return this.endVector(utf8.length);
  }

  /**
   * Executes `startObject` on `FlatBufferBuilder`.
   * @param numfields Parameter `numfields`.
   */
  public startObject(numfields: number): void {
    this.vtable = new Array(numfields).fill(0);
    this.objectStart = this.offset();
  }

  /**
   * Executes `addFieldInt8` on `FlatBufferBuilder`.
   * @param voffset Parameter `voffset`.
   * @param x Parameter `x`.
   * @param d Parameter `d`.
   */
  public addFieldInt8(voffset: number, x: number, d: number): void {
    if (x !== d) {
      this.addInt8(x);
      this.slot(voffset);
    }
  }

  /**
   * Executes `addFieldInt16` on `FlatBufferBuilder`.
   * @param voffset Parameter `voffset`.
   * @param x Parameter `x`.
   * @param d Parameter `d`.
   */
  public addFieldInt16(voffset: number, x: number, d: number): void {
    if (x !== d) {
      this.addInt16(x);
      this.slot(voffset);
    }
  }

  /**
   * Executes `addFieldInt32` on `FlatBufferBuilder`.
   * @param voffset Parameter `voffset`.
   * @param x Parameter `x`.
   * @param d Parameter `d`.
   */
  public addFieldInt32(voffset: number, x: number, d: number): void {
    if (x !== d) {
      this.addInt32(x);
      this.slot(voffset);
    }
  }

  /**
   * Executes `addFieldInt64` on `FlatBufferBuilder`.
   * @param voffset Parameter `voffset`.
   * @param x Parameter `x`.
   * @param d Parameter `d`.
   */
  public addFieldInt64(voffset: number, x: bigint, d: bigint): void {
    if (x !== d) {
      this.addInt64(x);
      this.slot(voffset);
    }
  }

  /**
   * Executes `addFieldFloat32` on `FlatBufferBuilder`.
   * @param voffset Parameter `voffset`.
   * @param x Parameter `x`.
   * @param d Parameter `d`.
   */
  public addFieldFloat32(voffset: number, x: number, d: number): void {
    if (x !== d) {
      this.addFloat32(x);
      this.slot(voffset);
    }
  }

  /**
   * Executes `addFieldFloat64` on `FlatBufferBuilder`.
   * @param voffset Parameter `voffset`.
   * @param x Parameter `x`.
   * @param d Parameter `d`.
   */
  public addFieldFloat64(voffset: number, x: number, d: number): void {
    if (x !== d) {
      this.addFloat64(x);
      this.slot(voffset);
    }
  }

  /**
   * Executes `addFieldOffset` on `FlatBufferBuilder`.
   * @param voffset Parameter `voffset`.
   * @param x Parameter `x`.
   * @param d Parameter `d`.
   */
  public addFieldOffset(voffset: number, x: number, d: number): void {
    if (x !== d) {
      this.addOffset(x);
      this.slot(voffset);
    }
  }

  /**
   * Executes `slot` on `FlatBufferBuilder`.
   * @param voffset Parameter `voffset`.
   */
  public slot(voffset: number): void {
    if (this.vtable !== null) {
      this.vtable[voffset] = this.offset();
    }
  }

  /**
   * Executes `endObject` on `FlatBufferBuilder`.
   * @returns Result of `endObject`.
   */
  public endObject(): number {
    if (this.vtable === null) {
      throw new Onnx9000Error('FlatBufferBuilder: endObject called without startObject');
    }
    this.addInt32(0); // placeholder for vtable offset
    const vtableloc = this.offset();

    // write vtable entries
    let i = this.vtable.length - 1;
    while (i >= 0 && this.vtable[i] === 0) {
      i--;
    }
    const trimmedSize = i + 1;

    for (let j = trimmedSize - 1; j >= 0; j--) {
      const vtableEntry = this.vtable[j];
      const off = vtableEntry ? vtableloc - vtableEntry : 0;
      this.addInt16(off);
    }

    const standardFields = 2; // vtable size, object size
    this.addInt16(vtableloc - this.objectStart);
    this.addInt16((trimmedSize + standardFields) * 2);

    let existingVtable = 0;
    const vt1 = this.space;
    const vt1len = this.view.getInt16(vt1, true);

    for (let j = 0; j < this.vtables.length; j++) {
      const vt2 = this.bb.length - (this.vtables[j] as number);
      const vt2len = this.view.getInt16(vt2, true);
      if (vt1len === vt2len) {
        let match = true;
        for (let k = 2; k < vt1len; k += 2) {
          if (this.view.getInt16(vt1 + k, true) !== this.view.getInt16(vt2 + k, true)) {
            match = false;
            break;
          }
        }
        if (match) {
          existingVtable = this.vtables[j] as number;
          break;
        }
      }
    }

    if (existingVtable) {
      this.space = this.bb.length - vtableloc; // remove vtable from buffer, keeping the 4-byte offset placeholder
      this.view.setInt32(this.space, existingVtable - vtableloc, true);
    } else {
      this.vtables.push(this.offset());
      this.view.setInt32(this.bb.length - vtableloc, this.offset() - vtableloc, true);
    }

    this.vtable = null;
    return vtableloc;
  }

  /**
   * Executes `finish` on `FlatBufferBuilder`.
   * @param rootTable Parameter `rootTable`.
   * @param identifier Parameter `identifier`.
   */
  public finish(rootTable: number, identifier?: string): void {
    this.prep(this.minalign, identifier ? 8 : 4);
    if (identifier) {
      const al = 4;
      this.prep(al, 4);
      for (let i = 3; i >= 0; i--) {
        this.addInt8(identifier.charCodeAt(i));
      }
    }
    this.addOffset(rootTable);
  }

  /**
   * Executes `offset` on `FlatBufferBuilder`.
   * @returns Result of `offset`.
   */
  public offset(): number {
    return this.bb.length - this.space;
  }

  /**
   * Executes `asUint8Array` on `FlatBufferBuilder`.
   * @returns Result of `asUint8Array`.
   */
  public asUint8Array(): Uint8Array {
    return this.bb.subarray(this.space, this.bb.length);
  }
}
