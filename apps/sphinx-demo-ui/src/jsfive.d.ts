/* eslint-disable */
// @ts-nocheck
declare module 'jsfive' {
  export class File {
    constructor(buffer: ArrayBuffer, name: string);
    get(name: string): object;
    keys: string[];
    attrs: object;
  }
  export interface Group {
    get(name: string): object;
    keys: string[];
    attrs: object;
  }
  export interface Dataset {
    shape: number[];
    dtype: string;
    value: object;
  }
}
