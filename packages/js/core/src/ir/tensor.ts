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
  name: string;
  shape: Shape;
  dtype: DType;
  isInitializer: boolean;
  requiresGrad: boolean;
  data: ArrayBufferView | null;

  constructor(
    name: string,
    shape: Shape,
    dtype: DType,
    isInitializer: boolean = false,
    requiresGrad: boolean = true,
    data: ArrayBufferView | null = null,
  ) {
    this.name = name;
    this.shape = shape;
    this.dtype = dtype;
    this.isInitializer = isInitializer;
    this.requiresGrad = requiresGrad;
    this.data = data;
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
}
