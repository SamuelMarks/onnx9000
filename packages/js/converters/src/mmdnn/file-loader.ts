/**
 * @fileoverview file-loader.ts
 * Provides file-loader functionality for the converters package.
 */
import { ConversionError } from '@onnx9000/core';
/** Represents the `FileLoader` class. */
export class FileLoader {
  /** Property `files` of `FileLoader`. */
  public files: Map<string, Blob>;

  /**
   * Initializes a new `FileLoader` instance.
   * @param files Parameter `files`.
   */
  constructor(files: (File | Blob)[]) {
    const validExtensions = [
      '.prototxt',
      '.caffemodel', // caffe
      '.json',
      '.params', // mxnet
      '.model', // cntk
      '.cfg',
      '.weights', // darknet
      '.param',
      '.bin', // ncnn
      '__model__',
      '.pdmodel',
      '.pdiparams', // paddle
      '.h5',
      '.keras', // keras
      '.mlmodel', // coreml
      '.zip',
      '.onnx', // onnx
      '.pbtxt', // tensorflow text proto
      '.pb', // tensorflow binary proto
      '.txt', // generic fallback
      '.py', // onnxscript
    ];

    for (const f of files) {
      if (f instanceof File) {
        if (!validExtensions.some((ext) => f.name.endsWith(ext)) && f.name !== '__model__') {
          throw new ConversionError(`Unsupported file type: ${f.name}`);
        }
      }
    }

    this.files = new Map();
    for (const f of files) {
      if (f instanceof File) {
        this.files.set(f.name, f);
      } else {
        // Blob without a filename, give it a default
        this.files.set(`blob_${Math.random().toString(36).slice(2)}`, f);
      }
    }
  }

  /** Executes `initialize` on `FileLoader`. */
  async initialize(): Promise<void> {
    // Initialization for any pre-fetching if necessary
  }

  /**
   * Executes `hasFile` on `FileLoader`.
   * @param name Parameter `name`.
   * @returns Result of `hasFile`.
   */
  hasFile(name: string): boolean {
    return this.files.has(name);
  }

  /**
   * Executes `getFile` on `FileLoader`.
   * @param name Parameter `name`.
   * @returns Result of `getFile`.
   */
  getFile(name: string): Blob {
    const file = this.files.get(name);
    if (!file) {
      throw new ConversionError(`File not found: ${name}`);
    }
    return file;
  }

  /**
   * Executes `readText` on `FileLoader`.
   * @param name Parameter `name`.
   * @returns Result of `readText`.
   */
  async readText(name: string): Promise<string> {
    const file = this.getFile(name);
    return file.text();
  }

  /**
   * Executes `readBuffer` on `FileLoader`.
   * @param name Parameter `name`.
   * @returns Result of `readBuffer`.
   */
  async readBuffer(name: string): Promise<ArrayBuffer> {
    const file = this.getFile(name);
    return file.arrayBuffer();
  }

  /**
   * Approximates memory-mapping for massive files by using Blob slicing.
   * Allows parsing massive binaries iteratively.
   */
  async readSlice(name: string, start: number, end: number): Promise<ArrayBuffer> {
    const file = this.getFile(name);
    const slice = file.slice(start, end);
    return slice.arrayBuffer();
  }

  /**
   * Executes `readSliceText` on `FileLoader`.
   * @param name Parameter `name`.
   * @param start Parameter `start`.
   * @param end Parameter `end`.
   * @returns Result of `readSliceText`.
   */
  async readSliceText(name: string, start: number, end: number): Promise<string> {
    const file = this.getFile(name);
    const slice = file.slice(start, end);
    return slice.text();
  }
}
