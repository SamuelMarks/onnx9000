import { Tensor, DType, SparseTensor } from './ir/tensor.js';

/**
 * Creates a typed array for the given DType and size.
 * @param dtype Internal DType string
 * @param size Number of elements
 * @returns A TypedArray (Float32Array, Int32Array, etc.)
 */
export function getTypedArray(dtype: DType, size: number): ArrayBufferView {
  switch (dtype) {
    case 'float32':
      return new Float32Array(size);
    case 'float64':
      return new Float64Array(size);
    case 'int8':
      return new Int8Array(size);
    case 'int16':
      return new Int16Array(size);
    case 'int32':
      return new Int32Array(size);
    case 'int64':
      return new BigInt64Array(size);
    case 'uint8':
      return new Uint8Array(size);
    case 'uint16':
      return new Uint16Array(size);
    case 'uint32':
      return new Uint32Array(size);
    case 'uint64':
      return new BigUint64Array(size);
    case 'bool':
      return new Uint8Array(size);
    case 'float16':
    case 'bfloat16':
      return new Uint16Array(size);
    default:
      console.warn(`Unsupported dtype for getTypedArray: ${dtype}, defaulting to float32`);
      return new Float32Array(size);
  }
}

/**
 * Unpacks tensor data into a standard JS array of numbers or bigints.
 * @param tensor The tensor to unpack
 * @returns An array containing the tensor's values
 */
export function unpackData(tensor: Tensor | null): (number | bigint)[] {
  if (!tensor || !tensor.data) return [];

  if (tensor.data instanceof Uint8Array) {
    if (tensor.dtype === 'float32') {
      const arr = new Float32Array(
        tensor.data.buffer,
        tensor.data.byteOffset,
        tensor.data.byteLength / 4,
      );
      return Array.from(arr);
    }

    if (tensor.dtype === 'int32') {
      const arr = new Int32Array(
        tensor.data.buffer,
        tensor.data.byteOffset,
        tensor.data.byteLength / 4,
      );
      return Array.from(arr);
    }

    if (tensor.dtype === 'int64') {
      const arr = new BigInt64Array(
        tensor.data.buffer,
        tensor.data.byteOffset,
        tensor.data.byteLength / 8,
      );
      return Array.from(arr);
    }

    const res: (number | bigint)[] = [];
    for (let i = 0; i < tensor.data.length; i++) res.push((tensor.data as Uint8Array)[i]!);
    return res;
  }

  if (Symbol.iterator in Object(tensor.data)) {
    return Array.from(tensor.data as ReturnType<typeof JSON.parse> as Iterable<number | bigint>);
  }

  return [];
}

/**
 * Converts a dense tensor to COO sparse format.
 * @param tensor Dense tensor to convert
 * @returns A SparseTensor in COO format
 */
export function denseToCoo(tensor: Tensor): SparseTensor {
  const values = unpackData(tensor);
  const nonZeroValues: (number | bigint)[] = [];
  const nonZeroIndices: number[] = [];

  for (let i = 0; i < values.length; i++) {
    const val = values[i];
    if (val !== 0 && val !== 0n) {
      nonZeroValues.push(val!);
      nonZeroIndices.push(i);
    }
  }

  const valData = getTypedArray(tensor.dtype, nonZeroValues.length);
  for (let i = 0; i < nonZeroValues.length; i++) {
    (valData as ReturnType<typeof JSON.parse>)[i] = nonZeroValues[i];
  }

  const valTensor = new Tensor(
    tensor.name + '_values',
    [nonZeroValues.length],
    tensor.dtype,
    true,
    false,
    valData as ArrayBufferView,
  );

  const idxData = new Int32Array(nonZeroIndices);
  const idxTensor = new Tensor(
    tensor.name + '_indices',
    [nonZeroIndices.length],
    'int32',
    true,
    false,
    new Uint8Array(idxData.buffer, idxData.byteOffset, idxData.byteLength),
  );

  return new SparseTensor(tensor.name, tensor.shape, 'COO', valTensor, idxTensor);
}

/**
 * Converts a dense tensor to CSR sparse format.
 * @param tensor Dense tensor to convert
 * @returns A SparseTensor in CSR format
 */
export function denseToCsr(tensor: Tensor): SparseTensor {
  // Simplified CSR: treats 2D tensors only
  const rows = (tensor.shape[0] as number) || 1;
  const cols = (tensor.shape[1] as number) || 1;
  const values = unpackData(tensor);

  const csrValues: (number | bigint)[] = [];
  const csrColIndices: number[] = [];
  const csrRowPtr: number[] = [0];
  const linearIndices: number[] = [];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = values[r * cols + c];
      if (val !== 0 && val !== 0n) {
        csrValues.push(val!);
        csrColIndices.push(c);
        linearIndices.push(r * cols + c);
      }
    }
    csrRowPtr.push(csrValues.length);
  }

  const valTensor = new Tensor(tensor.name + '_values', [csrValues.length], tensor.dtype, true);
  const vData = getTypedArray(tensor.dtype, csrValues.length);
  for (let i = 0; i < csrValues.length; i++)
    (vData as ReturnType<typeof JSON.parse>)[i] = csrValues[i];
  (valTensor as ReturnType<typeof JSON.parse>).data = vData;

  const colIdxTensor = new Tensor(
    tensor.name + '_col_indices',
    [csrColIndices.length],
    'int32',
    true,
  );
  const cData = new Int32Array(csrColIndices);
  (colIdxTensor as ReturnType<typeof JSON.parse>).data = new Uint8Array(
    cData.buffer,
    cData.byteOffset,
    cData.byteLength,
  );

  const rowPtrTensor = new Tensor(tensor.name + '_row_ptr', [csrRowPtr.length], 'int32', true);
  const rData = new Int32Array(csrRowPtr);
  (rowPtrTensor as ReturnType<typeof JSON.parse>).data = new Uint8Array(
    rData.buffer,
    rData.byteOffset,
    rData.byteLength,
  );

  const linearIdxTensor = new Tensor(
    tensor.name + '_indices',
    [linearIndices.length],
    'int32',
    true,
  );
  const lData = new Int32Array(linearIndices);
  (linearIdxTensor as ReturnType<typeof JSON.parse>).data = new Uint8Array(
    lData.buffer,
    lData.byteOffset,
    lData.byteLength,
  );

  return new SparseTensor(
    tensor.name,
    tensor.shape,
    'CSR',
    valTensor,
    linearIdxTensor,
    rowPtrTensor,
    colIdxTensor,
  );
}

/**
 * Converts a dense tensor to CSC sparse format.
 * @param tensor Dense tensor to convert
 * @returns A SparseTensor in CSC format
 */
export function denseToCsc(tensor: Tensor): SparseTensor {
  const rows = (tensor.shape[0] as number) || 1;
  const cols = (tensor.shape[1] as number) || 1;
  const values = unpackData(tensor);

  const cscValues: (number | bigint)[] = [];
  const cscRowIndices: number[] = [];
  const cscColPtr: number[] = [0];
  const linearIndices: number[] = [];

  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const val = values[r * cols + c];
      if (val !== 0 && val !== 0n) {
        cscValues.push(val!);
        cscRowIndices.push(r);
        linearIndices.push(r * cols + c);
      }
    }
    cscColPtr.push(cscValues.length);
  }

  const valTensor = new Tensor(tensor.name + '_values', [cscValues.length], tensor.dtype, true);
  const vData = getTypedArray(tensor.dtype, cscValues.length);
  for (let i = 0; i < cscValues.length; i++)
    (vData as ReturnType<typeof JSON.parse>)[i] = cscValues[i];
  (valTensor as ReturnType<typeof JSON.parse>).data = vData;

  const rowIdxTensor = new Tensor(
    tensor.name + '_row_indices',
    [cscRowIndices.length],
    'int32',
    true,
  );
  const rData = new Int32Array(cscRowIndices);
  (rowIdxTensor as ReturnType<typeof JSON.parse>).data = new Uint8Array(
    rData.buffer,
    rData.byteOffset,
    rData.byteLength,
  );

  const colPtrTensor = new Tensor(tensor.name + '_col_ptr', [cscColPtr.length], 'int32', true);
  const cData = new Int32Array(cscColPtr);
  (colPtrTensor as ReturnType<typeof JSON.parse>).data = new Uint8Array(
    cData.buffer,
    cData.byteOffset,
    cData.byteLength,
  );

  const linearIdxTensor = new Tensor(
    tensor.name + '_indices',
    [linearIndices.length],
    'int32',
    true,
  );
  const lData = new Int32Array(linearIndices);
  (linearIdxTensor as ReturnType<typeof JSON.parse>).data = new Uint8Array(
    lData.buffer,
    lData.byteOffset,
    lData.byteLength,
  );

  return new SparseTensor(
    tensor.name,
    tensor.shape,
    'CSC',
    valTensor,
    linearIdxTensor,
    colPtrTensor,
    rowIdxTensor,
  );
}

/**
 * Converts a dense tensor to BSR sparse format.
 * @param tensor Dense tensor to convert
 * @param blockDims Dimensions of the blocks
 * @returns A SparseTensor in BSR format
 */
export function denseToBsr(tensor: Tensor, blockDims: [number, number]): SparseTensor {
  const coo = denseToCoo(tensor);
  return new SparseTensor(
    tensor.name,
    tensor.shape,
    'BSR',
    coo.valuesTensor,
    coo.indicesTensor,
    null,
    null,
    blockDims,
  );
}

/**
 * Converts a sparse tensor to its COO representation.
 * @param sparseTensor The sparse tensor to convert
 * @returns A SparseTensor guaranteed to be in COO format
 */
export function sparseToCoo(sparseTensor: SparseTensor): SparseTensor {
  if (sparseTensor.format === 'COO') return sparseTensor;

  if (
    sparseTensor.format === 'CSR' ||
    sparseTensor.format === 'CSC' ||
    sparseTensor.format === 'BSR'
  ) {
    return new SparseTensor(
      sparseTensor.name,
      sparseTensor.shape,
      'COO',
      sparseTensor.valuesTensor,
      sparseTensor.indicesTensor,
    );
  }

  return sparseTensor;
}

/**
 * Converts a sparse tensor back into a dense representation.
 * @param sparseTensor The sparse tensor to convert
 * @returns A dense Tensor
 */
export function sparseToDense(sparseTensor: SparseTensor): Tensor {
  const coo = sparseToCoo(sparseTensor);
  if (!coo.valuesTensor || !coo.indicesTensor) {
    return new Tensor(coo.name, coo.shape, coo.dtype);
  }

  const values = unpackData(coo.valuesTensor);
  const indices = unpackData(coo.indicesTensor) as number[];

  let totalSize = 1;
  for (const dim of coo.shape) {
    if (typeof dim === 'number' && dim > 0) totalSize *= dim;
  }

  const denseData = getTypedArray(coo.dtype, totalSize);
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    if (idx !== undefined && idx >= 0 && idx < totalSize) {
      (denseData as ReturnType<typeof JSON.parse>)[idx] = values[i];
    }
  }

  return new Tensor(
    coo.name,
    coo.shape,
    coo.dtype,
    true,
    false,
    new Uint8Array(denseData.buffer, denseData.byteOffset, denseData.byteLength),
  );
}
