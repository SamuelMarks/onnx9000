import { Tensor, SparseTensor, Shape, DType } from './ir/tensor.js';

export function getTypedArray(dtype: DType, length: number): any {
  switch (dtype) {
    case 'float32':
      return new Float32Array(length);
    case 'float64':
      // Item 233: Handle explicit float64 fallback cleanly (downcasting to float32)
      console.warn('Downcasting float64 to float32 for web engine compatibility.');
      return new Float32Array(length);
    case 'int8':
      return new Int8Array(length);
    case 'int16':
      return new Int16Array(length);
    case 'int32':
      return new Int32Array(length);
    case 'uint8':
      return new Uint8Array(length);
    case 'uint16':
      return new Uint16Array(length);
    case 'uint32':
      return new Uint32Array(length);
    case 'int64':
      return new BigInt64Array(length);
    case 'uint64':
      return new BigUint64Array(length);
    case 'float16':
    case 'uint16':
      return new Uint16Array(length);
    case 'bool':
      return new Uint8Array(length);
    default:
      return new Float32Array(length);
  }
}

export function unpackData(tensor: Tensor): number[] | bigint[] {
  if (!tensor) return [];
  if (!tensor.data) return [];
  if (tensor.data instanceof Uint8Array) {
    const dv = new DataView(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength);
    const res: any[] = [];
    if (tensor.dtype === 'float32') {
      for (let i = 0; i < tensor.size; i++) res.push(dv.getFloat32(i * 4, true));
    } else if (tensor.dtype === 'int32') {
      for (let i = 0; i < tensor.size; i++) res.push(dv.getInt32(i * 4, true));
    } else if (tensor.dtype === 'int64') {
      for (let i = 0; i < tensor.size; i++) res.push(dv.getBigInt64(i * 8, true));
    } else {
      for (let i = 0; i < tensor.size; i++) res.push((tensor.data as any)[i]);
    }
    return res;
  }
  // @ts-ignore
  return Array.from(tensor.data);
}

export function denseToCoo(tensor: Tensor): SparseTensor {
  const values = unpackData(tensor);
  const nonZeroValues: any[] = [];
  const nonZeroIndices: number[] = [];

  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v !== 0 && v !== 0n) {
      nonZeroValues.push(v);
      nonZeroIndices.push(i);
    }
  }

  const valData = getTypedArray(tensor.dtype, nonZeroValues.length);
  for (let i = 0; i < nonZeroValues.length; i++) valData[i] = nonZeroValues[i];
  const valTensor = new Tensor(
    `${tensor.name}_values`,
    [nonZeroValues.length],
    tensor.dtype,
    true,
    false,
    valData,
  );

  const idxData = new Int32Array(nonZeroIndices);
  const idxTensor = new Tensor(
    `${tensor.name}_indices`,
    [nonZeroIndices.length],
    'int32',
    true,
    false,
    idxData,
  );

  return new SparseTensor(tensor.name, tensor.shape, 'COO', valTensor, idxTensor);
}

export function denseToCsr(tensor: Tensor): SparseTensor {
  if (tensor.shape.length !== 2) return denseToCoo(tensor);

  const rows = tensor.shape[0] as number;
  const cols = tensor.shape[1] as number;

  // Item 193: Throw warnings if a user attempts to sparsify a tiny model
  if (rows * cols < 1024) {
    console.warn(
      `Tensor '${tensor.name}' is small (${rows}x${cols}). CSR overhead might outweigh dense execution.`,
    );
  }

  // Item 195: Implement specific memory bounds checks preventing integer overflow
  if (rows > 2147483647 || cols > 2147483647) {
    throw new Error(`Tensor dimensions ${tensor.shape} exceed INT32 limits for CSR indexing.`);
  }

  const values = unpackData(tensor);

  const csrValues: any[] = [];
  const csrColIndices: number[] = [];
  const csrRowPtr: number[] = [0];

  for (let r = 0; r < rows; r++) {
    let count = 0;
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const v = values[idx];
      if (v !== 0 && v !== 0n) {
        csrValues.push(v);
        csrColIndices.push(c);
        count++;
      }
    }
    csrRowPtr.push(csrRowPtr[csrRowPtr.length - 1]! + count);
  }

  const valData = getTypedArray(tensor.dtype, csrValues.length);
  for (let i = 0; i < csrValues.length; i++) valData[i] = csrValues[i];
  const valTensor = new Tensor(
    `${tensor.name}_values`,
    [csrValues.length],
    tensor.dtype,
    true,
    false,
    valData,
  );

  const colIdxData = new Int32Array(csrColIndices);
  const colIdxTensor = new Tensor(
    `${tensor.name}_col_indices`,
    [csrColIndices.length],
    'int32',
    true,
    false,
    colIdxData,
  );

  const rowPtrData = new Int32Array(csrRowPtr);
  const rowPtrTensor = new Tensor(
    `${tensor.name}_row_ptr`,
    [csrRowPtr.length],
    'int32',
    true,
    false,
    rowPtrData,
  );

  return new SparseTensor(
    tensor.name,
    tensor.shape,
    'CSR',
    valTensor,
    null,
    rowPtrTensor,
    colIdxTensor,
  );
}

export function denseToCsc(tensor: Tensor): SparseTensor {
  if (tensor.shape.length !== 2) return denseToCoo(tensor);

  const rows = tensor.shape[0] as number;
  const cols = tensor.shape[1] as number;
  const values = unpackData(tensor);

  const cscValues: any[] = [];
  const cscRowIndices: number[] = [];
  const cscColPtr: number[] = [0];

  for (let c = 0; c < cols; c++) {
    let count = 0;
    for (let r = 0; r < rows; r++) {
      const idx = r * cols + c;
      const v = values[idx];
      if (v !== 0 && v !== 0n) {
        cscValues.push(v);
        cscRowIndices.push(r);
        count++;
      }
    }
    cscColPtr.push(cscColPtr[cscColPtr.length - 1]! + count);
  }

  const valData = getTypedArray(tensor.dtype, cscValues.length);
  for (let i = 0; i < cscValues.length; i++) valData[i] = cscValues[i];
  const valTensor = new Tensor(
    `${tensor.name}_values`,
    [cscValues.length],
    tensor.dtype,
    true,
    false,
    valData,
  );

  const rowIdxData = new Int32Array(cscRowIndices);
  const rowIdxTensor = new Tensor(
    `${tensor.name}_row_indices`,
    [cscRowIndices.length],
    'int32',
    true,
    false,
    rowIdxData,
  );

  const colPtrData = new Int32Array(cscColPtr);
  const colPtrTensor = new Tensor(
    `${tensor.name}_col_ptr`,
    [cscColPtr.length],
    'int32',
    true,
    false,
    colPtrData,
  );

  return new SparseTensor(
    tensor.name,
    tensor.shape,
    'CSC',
    valTensor,
    null,
    colPtrTensor,
    rowIdxTensor,
  );
}

export function denseToBsr(tensor: Tensor, blockDims: [number, number]): SparseTensor {
  if (tensor.shape.length !== 2) return denseToCoo(tensor);

  const rows = tensor.shape[0] as number;
  const cols = tensor.shape[1] as number;
  const bh = blockDims[0];
  const bw = blockDims[1];

  if (rows % bh !== 0 || cols % bw !== 0) return denseToCoo(tensor);

  const values = unpackData(tensor);
  const bsrValues: any[] = [];
  const bsrColIndices: number[] = [];
  const bsrRowPtr: number[] = [0];

  for (let rb = 0; rb < rows / bh; rb++) {
    let count = 0;
    for (let cb = 0; cb < cols / bw; cb++) {
      const block: any[] = [];
      let isNonZero = false;
      for (let lr = 0; lr < bh; lr++) {
        for (let lc = 0; lc < bw; lc++) {
          const v = values[(rb * bh + lr) * cols + (cb * bw + lc)];
          block.push(v);
          if (v !== 0 && v !== 0n) isNonZero = true;
        }
      }
      if (isNonZero) {
        bsrValues.push(...block);
        bsrColIndices.push(cb);
        count++;
      }
    }
    bsrRowPtr.push(bsrRowPtr[bsrRowPtr.length - 1]! + count);
  }

  const valData = getTypedArray(tensor.dtype, bsrValues.length);
  for (let i = 0; i < bsrValues.length; i++) valData[i] = bsrValues[i];
  const valTensor = new Tensor(
    `${tensor.name}_values`,
    [bsrValues.length],
    tensor.dtype,
    true,
    false,
    valData,
  );

  const colIdxData = new Int32Array(bsrColIndices);
  const colIdxTensor = new Tensor(
    `${tensor.name}_col_indices`,
    [bsrColIndices.length],
    'int32',
    true,
    false,
    colIdxData,
  );

  const rowPtrData = new Int32Array(bsrRowPtr);
  const rowPtrTensor = new Tensor(
    `${tensor.name}_row_ptr`,
    [bsrRowPtr.length],
    'int32',
    true,
    false,
    rowPtrData,
  );

  return new SparseTensor(
    tensor.name,
    tensor.shape,
    'BSR',
    valTensor,
    null,
    rowPtrTensor,
    colIdxTensor,
    blockDims,
  );
}

export function sparseToCoo(sparseTensor: SparseTensor): SparseTensor {
  if (sparseTensor.format === 'COO') return sparseTensor;

  const dims = sparseTensor.shape as number[];
  const rows = dims[0]!;
  const cols = dims[1]!;

  if (sparseTensor.format === 'CSR') {
    const csrValues = unpackData(sparseTensor.valuesTensor!);
    const csrColIndices = unpackData(sparseTensor.colIndicesTensor!) as number[];
    const csrRowPtr = unpackData(sparseTensor.rowPtrTensor!) as number[];

    const cooIndices: number[] = [];
    const cooValues: any[] = [];

    for (let r = 0; r < rows; r++) {
      for (let i = csrRowPtr[r]!; i < csrRowPtr[r + 1]!; i++) {
        const c = csrColIndices[i]!;
        cooIndices.push(r * cols + c);
        cooValues.push(csrValues[i]);
      }
    }

    const valData = getTypedArray(sparseTensor.dtype, cooValues.length);
    for (let i = 0; i < cooValues.length; i++) valData[i] = cooValues[i];
    const valTensor = new Tensor(
      `${sparseTensor.name}_values`,
      [cooValues.length],
      sparseTensor.dtype,
      true,
      false,
      valData,
    );

    const idxData = new Int32Array(cooIndices);
    const idxTensor = new Tensor(
      `${sparseTensor.name}_indices`,
      [cooIndices.length],
      'int32',
      true,
      false,
      idxData,
    );

    return new SparseTensor(sparseTensor.name, sparseTensor.shape, 'COO', valTensor, idxTensor);
  }

  if (sparseTensor.format === 'CSC') {
    const cscValues = unpackData(sparseTensor.valuesTensor!);
    const cscRowIndices = unpackData(sparseTensor.colIndicesTensor!) as number[];
    const cscColPtr = unpackData(sparseTensor.rowPtrTensor!) as number[];

    const cooIndices: number[] = [];
    const cooValues: any[] = [];

    for (let c = 0; c < cols; c++) {
      for (let i = cscColPtr[c]!; i < cscColPtr[c + 1]!; i++) {
        const r = cscRowIndices[i]!;
        cooIndices.push(r * cols + c);
        cooValues.push(cscValues[i]);
      }
    }

    const valData = getTypedArray(sparseTensor.valuesTensor!.dtype, cooValues.length);
    valData.set(cooValues);
    const valTensor = new Tensor(
      `${sparseTensor.name}_values`,
      [cooValues.length],
      sparseTensor.valuesTensor!.dtype,
      true,
      false,
      valData,
    );

    const idxData = new Int32Array(cooIndices);
    const idxTensor = new Tensor(
      `${sparseTensor.name}_indices`,
      [cooIndices.length],
      'int32',
      true,
      false,
      idxData,
    );

    return new SparseTensor(sparseTensor.name, sparseTensor.shape, 'COO', valTensor, idxTensor);
  }

  if (sparseTensor.format === 'BSR') {
    const bsrValues = unpackData(sparseTensor.valuesTensor!);
    const bsrColIndices = unpackData(sparseTensor.colIndicesTensor!) as number[];
    const bsrRowPtr = unpackData(sparseTensor.rowPtrTensor!) as number[];
    const [bh, bw] = sparseTensor.blockDims!;

    const cooIndices: number[] = [];
    const cooValues: any[] = [];

    let blockIdx = 0;
    const rbCount = Math.floor(rows / bh);
    for (let rb = 0; rb < rbCount; rb++) {
      for (let i = bsrRowPtr[rb]!; i < bsrRowPtr[rb + 1]!; i++) {
        const cb = bsrColIndices[i]!;
        for (let r = 0; r < bh; r++) {
          for (let c = 0; c < bw; c++) {
            const val = bsrValues[blockIdx * bh * bw + r * bw + c];
            if (val !== 0) {
              cooIndices.push((rb * bh + r) * cols + (cb * bw + c));
              cooValues.push(val);
            }
          }
        }
        blockIdx++;
      }
    }

    const valData = getTypedArray(sparseTensor.valuesTensor!.dtype, cooValues.length);
    valData.set(cooValues);
    const valTensor = new Tensor(
      `${sparseTensor.name}_values`,
      [cooValues.length],
      sparseTensor.valuesTensor!.dtype,
      true,
      false,
      valData,
    );

    const idxData = new Int32Array(cooIndices);
    const idxTensor = new Tensor(
      `${sparseTensor.name}_indices`,
      [cooIndices.length],
      'int32',
      true,
      false,
      idxData,
    );

    return new SparseTensor(sparseTensor.name, sparseTensor.shape, 'COO', valTensor, idxTensor);
  }

  return sparseTensor;
}

export function sparseToDense(sparseTensor: SparseTensor): Tensor {
  const coo = sparseToCoo(sparseTensor);
  if (!coo.valuesTensor || !coo.indicesTensor) {
    return new Tensor(coo.name, coo.shape, coo.dtype);
  }

  const values = unpackData(coo.valuesTensor);
  const indices = unpackData(coo.indicesTensor) as number[];

  let totalSize = 1;
  for (const dim of coo.shape) {
    if (typeof dim === 'number') totalSize *= dim;
  }

  const denseData = getTypedArray(coo.dtype, totalSize);
  for (let i = 0; i < indices.length; i++) {
    denseData[indices[i]!] = values[i];
  }

  return new Tensor(coo.name, coo.shape, coo.dtype, true, false, denseData);
}
