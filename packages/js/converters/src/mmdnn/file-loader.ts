/* eslint-disable */
// @ts-nocheck
export class FileLoader {
  private files: Map<string, Blob>;

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
          throw new Error(`Unsupported file type: ${f.name}`);
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

  async initialize(): Promise<void> {
    // Initialization for any pre-fetching if necessary
  }

  hasFile(name: string): boolean {
    return this.files.has(name);
  }

  getFile(name: string): Blob {
    const file = this.files.get(name);
    if (!file) {
      throw new Error(`File not found: ${name}`);
    }
    return file;
  }

  async readText(name: string): Promise<string> {
    const file = this.getFile(name);
    return file.text();
  }

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

  async readSliceText(name: string, start: number, end: number): Promise<string> {
    const file = this.getFile(name);
    const slice = file.slice(start, end);
    return slice.text();
  }
}
