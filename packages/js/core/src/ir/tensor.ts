export type DType =
  | 'float32'
  | 'float64'
  | 'int8'
  | 'int16'
  | 'int32'
  | 'int64'
  | 'uint8'
  | 'uint16'
  | 'uint32'
  | 'uint64'
  | 'bool'
  | 'string'
  | 'float16'
  | 'bfloat16';

export type DynamicDim = string | -1;
export type Shape = (number | DynamicDim)[];

export class Tensor {
  id: string;
  name: string;
  shape: Shape;
  dtype: DType;
  isInitializer: boolean;
  requiresGrad: boolean;
  data: ArrayBufferView | null;
  externalData?: { location: string; offset: number; length: number } | undefined;

  constructor(
    name: string,
    shape: Shape,
    dtype: DType,
    isInitializer: boolean = false,
    requiresGrad: boolean = true,
    data: ArrayBufferView | null = null,
    externalData?: { location: string; offset: number; length: number },
  ) {
    this.id =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : Math.random().toString(36).substring(2);
    this.name = name;
    this.shape = shape;
    this.dtype = dtype;
    this.isInitializer = isInitializer;
    this.requiresGrad = requiresGrad;
    this.data = data;
    if (externalData) {
      this.externalData = externalData;
    }
  }

  get size(): number {
    let s = 1;
    for (const dim of this.shape) {
      if (typeof dim === 'number' && dim > 0) {
        s *= dim;
      }
    }
    return s;
  }

  formatData(limit: number = 100): string {
    if (!this.data) return 'No data';
    const values: number[] = [];
    const maxVals = Math.min(this.size, limit);
    // basic assumption: Float32Array or similar
    if (this.data instanceof Uint8Array) {
      const dv = new DataView(this.data.buffer, this.data.byteOffset, this.data.byteLength);
      // if we just have raw bytes
      if (this.dtype === 'float32') {
        for (let i = 0; i < maxVals; i++) values.push(dv.getFloat32(i * 4, true));
      } else if (this.dtype === 'int64') {
        // handle int64
        for (let i = 0; i < maxVals; i++) values.push(Number(dv.getBigInt64(i * 8, true)));
      } else if (this.dtype === 'float16') {
        // approximate decoding float16
        for (let i = 0; i < maxVals; i++) {
          const val = dv.getUint16(i * 2, true);
          // ignoring subnormals, inf, nan for simplicity of formatting string
          const sign = (val & 0x8000) >> 15;
          const exp = (val & 0x7c00) >> 10;
          const frac = val & 0x03ff;
          values.push(exp === 0 ? 0 : (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024));
        }
      } else if (this.dtype === 'bfloat16') {
        for (let i = 0; i < maxVals; i++) {
          const val = dv.getUint16(i * 2, true);
          // bfloat16 is essentially float32 with truncated mantissa
          const tmp = new Uint32Array(1);
          tmp[0] = val << 16;
          const f32 = new Float32Array(tmp.buffer);
          values.push(f32[0]!);
        }
      } else {
        for (let i = 0; i < maxVals; i++) values.push(this.data[i]!);
      }
    } else {
      for (let i = 0; i < maxVals; i++) {
        // @ts-ignore
        values.push(this.data[i]);
      }
    }

    if (this.size > maxVals) {
      return `[${values.join(', ')} ... +${this.size - maxVals} elements]`;
    }
    return `[${values.join(', ')}]`;
  }

  copy(): Tensor {
    if (this instanceof SparseTensor) {
      return new SparseTensor(
        this.name,
        [...this.shape],
        this.format,
        this.valuesTensor?.copy() || null,
        this.indicesTensor?.copy() || null,
        this.rowPtrTensor?.copy() || null,
        this.colIndicesTensor?.copy() || null,
        this.blockDims ? [...this.blockDims] : undefined,
      );
    }
    const t = new Tensor(
      this.name,
      [...this.shape],
      this.dtype,
      this.isInitializer,
      this.requiresGrad,
    );
    if (this.data) {
      // Shallow copy of view or deep copy of buffer?
      // For IR, usually we might want to copy the data too if it's being mutated.
      // But let's keep it simple for now.
      t.data = this.data;
    }
    if (this.externalData) {
      t.externalData = { ...this.externalData };
    }
    return t;
  }
}

export type SparseFormat = 'COO' | 'CSR' | 'CSC' | 'BSR';

export class SparseTensor extends Tensor {
  valuesTensor: Tensor | null;
  indicesTensor: Tensor | null;
  rowPtrTensor: Tensor | null;
  colIndicesTensor: Tensor | null;
  format: SparseFormat;
  blockDims?: [number, number];

  constructor(
    name: string,
    shape: Shape,
    format: SparseFormat = 'COO',
    valuesTensor: Tensor | null = null,
    indicesTensor: Tensor | null = null,
    rowPtrTensor: Tensor | null = null,
    colIndicesTensor: Tensor | null = null,
    blockDims?: [number, number],
  ) {
    super(name, shape, valuesTensor?.dtype || 'float32', true);
    this.format = format;
    this.valuesTensor = valuesTensor;
    this.indicesTensor = indicesTensor;
    this.rowPtrTensor = rowPtrTensor;
    this.colIndicesTensor = colIndicesTensor;
    if (blockDims !== undefined) {
      this.blockDims = blockDims;
    }
  }
}
