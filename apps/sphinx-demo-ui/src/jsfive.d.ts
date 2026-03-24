declare module 'jsfive' {
  export class File {
    constructor(buffer: ArrayBuffer, name: string);
    get(name: string): any;
    keys: string[];
    attrs: any;
  }
  export interface Group {
    get(name: string): any;
    keys: string[];
    attrs: any;
  }
  export interface Dataset {
    shape: number[];
    dtype: string;
    value: any;
  }
}
