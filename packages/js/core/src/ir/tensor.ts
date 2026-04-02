/**
 * Supported ONNX data types in onnx9000.
 */
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

/**
 * Represents a dynamic dimension, which can be a symbolic name or -1.
 */
export type DynamicDim = string | -1;

/**
 * Represents the shape of a tensor as an array of dimensions.
 */
export type Shape = (number | DynamicDim)[];

/**
 * Internal Representation of a Tensor (Edge).
 */
export class Tensor {
  /** Unique identifier for the tensor instance. */
  id: string;
  /** Name of the tensor in the graph. */
  name: string;
  /** Shape of the tensor. */
  shape: Shape;
  /** Data type of the tensor. */
  dtype: DType;
  /** Whether this tensor is a constant initializer. */
  isInitializer: boolean;
  /** Whether to track gradients for this tensor (for training). */
  requiresGrad: boolean;
  /** The actual data buffer view, if loaded. */
  data: ArrayBufferView | null;
  /** Optional metadata for externally stored data. */
  externalData?: { location: string; offset: number; length: number } | undefined;

  /**
   * Create a new Tensor.
   * @param name Name of the tensor.
   * @param shape Shape of the tensor.
   * @param dtype Data type of the tensor.
   * @param isInitializer Whether it's an initializer.
   * @param requiresGrad Whether it requires gradient.
   * @param data Optional data buffer.
   * @param externalData Optional external data metadata.
   */
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

  /**
   * Get the total number of elements in the tensor.
   */
  get size(): number {
    let s = 1;
    for (const dim of this.shape) {
      if (typeof dim === 'number' && dim > 0) {
        s *= dim;
      }
    }
    return s;
  }

  /**
   * Format the tensor data as a human-readable string.
   * @param limit Maximum number of elements to include.
   * @returns A formatted string representing the tensor data.
   */
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

  /**
   * Create a deep copy of the tensor.
   * @returns A new Tensor instance with copied metadata.
   */
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

/**
 * Storage formats for Sparse Tensors.
 */
export type SparseFormat = 'COO' | 'CSR' | 'CSC' | 'BSR';

/**
 * Internal Representation of a Sparse Tensor.
 */
export class SparseTensor extends Tensor {
  /** Tensor containing the non-zero values. */
  valuesTensor: Tensor | null;
  /** Tensor containing the indices. */
  indicesTensor: Tensor | null;
  /** Tensor containing the row pointers (for CSR/CSC). */
  rowPtrTensor: Tensor | null;
  /** Tensor containing the column indices (for CSR/CSC). */
  colIndicesTensor: Tensor | null;
  /** The storage format used. */
  format: SparseFormat;
  /** Block dimensions for BSR format. */
  blockDims?: [number, number];

  /**
   * Create a new SparseTensor.
   * @param name Name of the tensor.
   * @param shape Logical shape.
   * @param format Storage format.
   * @param valuesTensor Values tensor.
   * @param indicesTensor Indices tensor.
   * @param rowPtrTensor Row pointers tensor.
   * @param colIndicesTensor Column indices tensor.
   * @param blockDims Block dimensions.
   */
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
