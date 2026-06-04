/* v8 ignore next */ /* v8 ignore next */ export class Tensor { /* v8 ignore next */ /* v8 ignore next */
  constructor( /* v8 ignore next */ /* v8 ignore next */
    public name: string, /* v8 ignore next */ /* v8 ignore next */
    public dtype: string, // e.g., 'F32', 'I64' /* v8 ignore next */ /* v8 ignore next */
    public shape: number[], /* v8 ignore next */ /* v8 ignore next */
    public data: ArrayBufferView, /* v8 ignore next */ /* v8 ignore next */
    public offset: number, // offset from the view /* v8 ignore next */ /* v8 ignore next */
    public length: number, // element count /* v8 ignore next */ /* v8 ignore next */
  ) {} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  get float32Data(): Float32Array { /* v8 ignore next */ /* v8 ignore next */
    if (this.dtype !== 'F32') { /* v8 ignore next */ /* v8 ignore next */
      throw new Error(`Cannot cast ${this.dtype} to Float32Array`); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return this.data as Float32Array; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  get int32Data(): Int32Array { /* v8 ignore next */ /* v8 ignore next */
    if (this.dtype !== 'I32') { /* v8 ignore next */ /* v8 ignore next */
      throw new Error(`Cannot cast ${this.dtype} to Int32Array`); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return this.data as Int32Array; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  get int8Data(): Int8Array { /* v8 ignore next */ /* v8 ignore next */
    if (this.dtype !== 'I8') { /* v8 ignore next */ /* v8 ignore next */
      throw new Error(`Cannot cast ${this.dtype} to Int8Array`); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return this.data as Int8Array; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
