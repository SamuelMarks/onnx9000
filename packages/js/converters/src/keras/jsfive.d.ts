declare module 'jsfive' {
  export class File {
    constructor(buffer: ArrayBuffer, filename: string);
    keys: string[];
    get(path: string): Group | Dataset;
  }

  export interface Group {
    keys: string[];
    attrs: Record<string, string | number | number[] | string[] | Uint8Array>;
    get(path: string): Group | Dataset;
  }

  export interface Dataset {
    shape: number[];
    dtype: string;
    value: Float32Array | Int32Array | Uint8Array | Float64Array;
    attrs: Record<string, string | number | number[] | string[] | Uint8Array>;
  }
}
