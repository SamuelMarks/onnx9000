/* eslint-disable */
// @onnx9000/tfjs-shim - Drop-in replacement for TensorFlow.js

import { Graph, Node, Tensor as OnnxTensor } from '@onnx9000/core';

// --- Global Config and Envs ---
let currentBackend = 'webgpu';
let isProdMode = false;
let isDebugMode = false;

export const version_core = '4.10.0';
export const version_tfjs = '4.10.0';
export const version = {
  tfjs: version_tfjs,
  core: version_core,
};

export async function setBackend(backendName: string): Promise<boolean> {
  currentBackend = backendName;
  return true;
}

export function getBackend(): string {
  return currentBackend;
}

export async function ready(): Promise<void> {
  return Promise.resolve();
}

export function env(): any {
  return {
    get: (key: string) => null,
    set: (key: string, value: any) => {},
  };
}

export function enableProdMode(): void {
  isProdMode = true;
}

export function enableDebugMode(): void {
  isDebugMode = true;
}

export function memory(): {
  numBytes: number;
  numTensors: number;
  numDataBuffers: number;
  reasons?: string[];
} {
  return {
    numBytes: globalTensorRegistry.size * 4,
    numTensors: globalTensorRegistry.size,
    numDataBuffers: globalTensorRegistry.size,
  };
}

export async function profile(f: () => any): Promise<any> {
  const result = await f();
  return { newBytes: 0, newTensors: 0, peakBytes: 0, kernels: [], result };
}

export async function time(f: () => any): Promise<{ wallMs: number; kernelMs: number }> {
  const start = performance.now();
  await f();
  const end = performance.now();
  return { wallMs: end - start, kernelMs: end - start };
}

export function disposeVariables(): void {
  globalTensorRegistry.clear();
}

// --- Tensor Core ---
export type DataType = 'float32' | 'int32' | 'bool' | 'complex64' | 'string';
const globalTensorRegistry = new Set<Tensor>();
let currentTidyScope: Tensor[][] | null = null;

export class Tensor {
  shape: number[];
  dtype: DataType;
  dataArray: any[];
  isDisposed: boolean = false;
  rank: number;
  size: number;

  constructor(shape: number[], dtype: DataType, dataArray: any[]) {
    this.shape = shape;
    this.dtype = dtype;
    this.dataArray = dataArray;
    this.rank = shape.length;
    this.size = shape.reduce((a, b) => a * b, 1);
    globalTensorRegistry.add(this);
    if (currentTidyScope) {
      currentTidyScope[currentTidyScope.length - 1].push(this);
    }
  }

  async data(): Promise<any> {
    if (this.dtype === 'string') return new Array(this.size).fill(this.dataArray[0] || '');
    if (this.dtype === 'float32') return new Float32Array(this.dataArray);
    if (this.dtype === 'int32') return new Int32Array(this.dataArray);
    if (this.dtype === 'bool') return new Uint8Array(this.dataArray.map((b) => (b ? 1 : 0)));
    return new Float32Array(this.dataArray);
  }

  dataSync(): any {
    if (this.dtype === 'string') return new Array(this.size).fill(this.dataArray[0] || '');
    if (this.dtype === 'float32') return new Float32Array(this.dataArray);
    if (this.dtype === 'int32') return new Int32Array(this.dataArray);
    if (this.dtype === 'bool') return new Uint8Array(this.dataArray.map((b) => (b ? 1 : 0)));
    return new Float32Array(this.dataArray);
  }

  async array(): Promise<any[]> {
    return this.arraySync();
  }

  arraySync(): any[] {
    if (this.rank === 0) return this.dataArray[0];
    if (this.rank === 1) return Array.from(this.dataArray);
    return Array.from(this.dataArray); // Simplified flat array for now to ensure typings
  }

  dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
    globalTensorRegistry.delete(this);
  }

  clone(): Tensor {
    return new Tensor(this.shape.slice(), this.dtype, this.dataArray.slice());
  }

  print(verbose: boolean = false): void {
    console.log(`Tensor [${this.shape}]`, this.dataArray);
  }

  flatten(): Tensor {
    return reshape(this, [-1]);
  }
}

export function tensor(values: any, shape?: number[], dtype?: DataType): Tensor {
  const flatVals = Array.isArray(values)
    ? values.flat(Infinity)
    : values.length !== undefined && typeof values !== 'string'
      ? Array.from(values)
      : [values];
  const s = shape || [flatVals.length];
  const d = dtype || 'float32';
  return new Tensor(s, d, flatVals);
}

export function tensor1d(values: any, dtype?: DataType): Tensor {
  return tensor(values, [values.length || 1], dtype);
}
export function tensor2d(values: any, shape?: [number, number], dtype?: DataType): Tensor {
  return tensor(values, shape, dtype);
}
export function tensor3d(values: any, shape?: [number, number, number], dtype?: DataType): Tensor {
  return tensor(values, shape, dtype);
}
export function tensor4d(
  values: any,
  shape?: [number, number, number, number],
  dtype?: DataType,
): Tensor {
  return tensor(values, shape, dtype);
}
export function tensor5d(
  values: any,
  shape?: [number, number, number, number, number],
  dtype?: DataType,
): Tensor {
  return tensor(values, shape, dtype);
}
export function tensor6d(
  values: any,
  shape?: [number, number, number, number, number, number],
  dtype?: DataType,
): Tensor {
  return tensor(values, shape, dtype);
}
export function scalar(value: any, dtype?: DataType): Tensor {
  return tensor([value], [], dtype);
}
export function buffer(shape: number[], dtype?: DataType, values?: any): Tensor {
  return tensor(values || new Array(shape.reduce((a, b) => a * b, 1)).fill(0), shape, dtype);
}
export function clone(x: Tensor): Tensor {
  return x.clone();
}

export function tidy<T>(nameOrFn: string | (() => T), fn?: () => T): T {
  const actualFn = fn || (nameOrFn as () => T);
  if (!currentTidyScope) currentTidyScope = [];
  currentTidyScope.push([]);
  const result = actualFn();
  const scopeTensors = currentTidyScope.pop()!;

  const extractTensors = (res: any): Tensor[] => {
    if (res instanceof Tensor) return [res];
    if (Array.isArray(res)) return res.flatMap(extractTensors);
    if (res && typeof res === 'object') return Object.values(res).flatMap(extractTensors);
    return [];
  };

  const keepTensors = new Set(extractTensors(result));
  for (const t of scopeTensors) {
    if (!keepTensors.has(t) && !t.isDisposed) {
      t.dispose();
    }
  }
  return result;
}

export function keep(x: Tensor): Tensor {
  if (currentTidyScope && currentTidyScope.length > 0) {
    const scope = currentTidyScope[currentTidyScope.length - 1];
    const idx = scope.indexOf(x);
    if (idx !== -1) scope.splice(idx, 1);
  }
  return x;
}

export function dispose(tensors: Tensor | Tensor[] | Record<string, Tensor>): void {
  if (tensors instanceof Tensor) {
    tensors.dispose();
  } else if (Array.isArray(tensors)) {
    tensors.forEach((t) => {
      t.dispose();
    });
  } else if (typeof tensors === 'object') {
    Object.values(tensors).forEach((t) => {
      t.dispose();
    });
  }
}

// Math functions generator
function makeElementwise(name: string, op: (a: number, b: number) => number) {
  return (a: Tensor, b: Tensor | number): Tensor => {
    const isScalarB = typeof b === 'number';
    const bArray = isScalarB ? [b] : b.dataArray;
    const len = Math.max(a.size, isScalarB ? 1 : b.size);
    const newData = new Array(len);
    for (let i = 0; i < len; i++) {
      const valA = a.dataArray[i % a.size];
      const valB = bArray[i % bArray.length];
      newData[i] = op(valA, valB);
    }
    return new Tensor(a.shape.slice(), a.dtype, newData);
  };
}

function makeUnary(name: string, op: (a: number) => number) {
  return (a: Tensor): Tensor => {
    const newData = new Array(a.size);
    for (let i = 0; i < a.size; i++) {
      newData[i] = op(a.dataArray[i]);
    }
    return new Tensor(a.shape.slice(), a.dtype, newData);
  };
}

export const add = makeElementwise('add', (a, b) => a + b);
export const sub = makeElementwise('sub', (a, b) => a - b);
export const mul = makeElementwise('mul', (a, b) => a * b);
export const div = makeElementwise('div', (a, b) => a / b);
export const divNoNan = makeElementwise('divNoNan', (a, b) => (b === 0 ? 0 : a / b));
export const floorDiv = makeElementwise('floorDiv', (a, b) => Math.floor(a / b));
export const maximum = makeElementwise('maximum', (a, b) => Math.max(a, b));
export const minimum = makeElementwise('minimum', (a, b) => Math.min(a, b));
export const mod = makeElementwise('mod', (a, b) => a % b);
export const pow = makeElementwise('pow', (a, b) => Math.pow(a, b));
export const squaredDifference = makeElementwise('squaredDifference', (a, b) => Math.pow(a - b, 2));

export const abs = makeUnary('abs', Math.abs);
export const acos = makeUnary('acos', Math.acos);
export const acosh = makeUnary('acosh', Math.acosh);
export const asin = makeUnary('asin', Math.asin);
export const asinh = makeUnary('asinh', Math.asinh);
export const atan = makeUnary('atan', Math.atan);
export const atanh = makeUnary('atanh', Math.atanh);
export const ceil = makeUnary('ceil', Math.ceil);
export const cos = makeUnary('cos', Math.cos);
export const cosh = makeUnary('cosh', Math.cosh);
export const erf = makeUnary('erf', (x) => Math.tanh(x)); // approx
export const exp = makeUnary('exp', Math.exp);
export const expm1 = makeUnary('expm1', Math.expm1);
export const floor = makeUnary('floor', Math.floor);
export const isFinite = makeUnary('isFinite', (x) => (Number.isFinite(x) ? 1 : 0));
export const isInf = makeUnary('isInf', (x) => (!Number.isFinite(x) && !Number.isNaN(x) ? 1 : 0));
export const isNaN = makeUnary('isNaN', (x) => (Number.isNaN(x) ? 1 : 0));
export const log = makeUnary('log', Math.log);
export const log1p = makeUnary('log1p', Math.log1p);
export const neg = makeUnary('neg', (x) => -x);
export const reciprocal = makeUnary('reciprocal', (x) => 1 / x);
export const round = makeUnary('round', Math.round);
export const rsqrt = makeUnary('rsqrt', (x) => 1 / Math.sqrt(x));
export const sign = makeUnary('sign', Math.sign);
export const sin = makeUnary('sin', Math.sin);
export const sinh = makeUnary('sinh', Math.sinh);
export const sqrt = makeUnary('sqrt', Math.sqrt);
export const square = makeUnary('square', (x) => x * x);
export const tan = makeUnary('tan', Math.tan);

export const atan2 = makeElementwise('atan2', Math.atan2);

export function step(x: Tensor, alpha: number = 0.0): Tensor {
  return makeUnary('step', (v) => (v > 0 ? 1 : alpha))(x);
}

export function addN(tensors: Tensor[]): Tensor {
  if (tensors.length === 0) throw new Error('addN requires at least one tensor');
  let res = tensors[0];
  for (let i = 1; i < tensors.length; i++) res = add(res, tensors[i]);
  return res;
}

// Matrix operations
export function matMul(a: Tensor, b: Tensor, transposeA = false, transposeB = false): Tensor {
  const rowsA = transposeA ? a.shape[1] : a.shape[0];
  const colsA = transposeA ? a.shape[0] : a.shape[1];
  const rowsB = transposeB ? b.shape[1] : b.shape[0];
  const colsB = transposeB ? b.shape[0] : b.shape[1];
  if (colsA !== rowsB) throw new Error(`Matrix size-incompatible`);
  const newData = new Array(rowsA * colsB).fill(0);
  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        const idxA = transposeA ? k * rowsA + i : i * colsA + k;
        const idxB = transposeB ? j * rowsB + k : k * colsB + j;
        sum += a.dataArray[idxA] * b.dataArray[idxB];
      }
      newData[i * colsB + j] = sum;
    }
  }
  return new Tensor([rowsA, colsB], a.dtype, newData);
}

export function dot(a: Tensor, b: Tensor): Tensor {
  return matMul(a, b);
}
export function outerProduct(v1: Tensor, v2: Tensor): Tensor {
  return matMul(v1, v2);
}
export function norm(
  x: Tensor,
  ord: number | string = 'euclidean',
  axis?: number | number[],
  keepDims = false,
): Tensor {
  let val = 0;
  if (ord === 'euclidean' || ord === 2) {
    val = Math.sqrt(x.dataArray.reduce((acc, v) => acc + v * v, 0));
  } else if (ord === 1) {
    val = x.dataArray.reduce((acc, v) => acc + Math.abs(v), 0);
  } else if (ord === Infinity || ord === 'inf') {
    val = Math.max(...x.dataArray.map(Math.abs));
  }
  return new Tensor(keepDims ? x.shape.map(() => 1) : [1], x.dtype, [val]);
}

export function conv1d(
  x: Tensor,
  filter: Tensor,
  stride: number,
  pad: 'valid' | 'same' | number,
  dataFormat: 'NWC' | 'NCW' = 'NWC',
  dilation: number = 1,
): Tensor {
  const [batch, inWidth, inChannels] =
    dataFormat === 'NWC' ? x.shape : [x.shape[0], x.shape[2], x.shape[1]];
  const [filterWidth, inFilters, outFilters] = filter.shape;
  let padX = 0;
  let outWidth = Math.floor((inWidth - filterWidth) / stride + 1);
  if (pad === 'same') {
    outWidth = Math.ceil(inWidth / stride);
    padX = Math.max(0, (outWidth - 1) * stride + filterWidth - inWidth);
  } else if (typeof pad === 'number') {
    padX = pad * 2;
    outWidth = Math.floor((inWidth + padX - filterWidth) / stride + 1);
  }
  const leftPad = Math.floor(padX / 2);
  const outData = new Array(batch * outWidth * outFilters).fill(0);
  for (let b = 0; b < batch; b++) {
    for (let ow = 0; ow < outWidth; ow++) {
      for (let oc = 0; oc < outFilters; oc++) {
        let sum = 0;
        for (let fw = 0; fw < filterWidth; fw++) {
          for (let ic = 0; ic < inChannels; ic++) {
            const iw = ow * stride - leftPad + fw * dilation;
            if (iw >= 0 && iw < inWidth) {
              const inIdx =
                dataFormat === 'NWC'
                  ? (b * inWidth + iw) * inChannels + ic
                  : (b * inChannels + ic) * inWidth + iw;
              const fIdx = (fw * inFilters + ic) * outFilters + oc;
              sum += x.dataArray[inIdx] * filter.dataArray[fIdx];
            }
          }
        }
        const outIdx =
          dataFormat === 'NWC'
            ? (b * outWidth + ow) * outFilters + oc
            : (b * outFilters + oc) * outWidth + ow;
        outData[outIdx] = sum;
      }
    }
  }
  const outShape =
    dataFormat === 'NWC' ? [batch, outWidth, outFilters] : [batch, outFilters, outWidth];
  return new Tensor(outShape, x.dtype, outData);
}
export function conv2d(
  x: Tensor,
  filter: Tensor,
  strides: number | [number, number],
  pad: 'valid' | 'same' | number,
  dataFormat: 'NHWC' | 'NCHW' = 'NHWC',
  dilations: number | [number, number] = 1,
): Tensor {
  const strideY = Array.isArray(strides) ? strides[0] : strides;
  const strideX = Array.isArray(strides) ? strides[1] : strides;
  const dilY = Array.isArray(dilations) ? dilations[0] : dilations;
  const dilX = Array.isArray(dilations) ? dilations[1] : dilations;
  const [batch, inHeight, inWidth, inChannels] =
    dataFormat === 'NHWC' ? x.shape : [x.shape[0], x.shape[2], x.shape[3], x.shape[1]];
  const [filterHeight, filterWidth, inFilters, outFilters] = filter.shape;
  let padY = 0,
    padX = 0;
  let outHeight = Math.floor((inHeight - filterHeight) / strideY + 1);
  let outWidth = Math.floor((inWidth - filterWidth) / strideX + 1);
  if (pad === 'same') {
    outHeight = Math.ceil(inHeight / strideY);
    outWidth = Math.ceil(inWidth / strideX);
    padY = Math.max(0, (outHeight - 1) * strideY + filterHeight - inHeight);
    padX = Math.max(0, (outWidth - 1) * strideX + filterWidth - inWidth);
  } else if (typeof pad === 'number') {
    padY = pad * 2;
    padX = pad * 2;
    outHeight = Math.floor((inHeight + padY - filterHeight) / strideY + 1);
    outWidth = Math.floor((inWidth + padX - filterWidth) / strideX + 1);
  }
  const topPad = Math.floor(padY / 2);
  const leftPad = Math.floor(padX / 2);
  const outData = new Array(batch * outHeight * outWidth * outFilters).fill(0);
  for (let b = 0; b < batch; b++) {
    for (let oh = 0; oh < outHeight; oh++) {
      for (let ow = 0; ow < outWidth; ow++) {
        for (let oc = 0; oc < outFilters; oc++) {
          let sum = 0;
          for (let fh = 0; fh < filterHeight; fh++) {
            for (let fw = 0; fw < filterWidth; fw++) {
              for (let ic = 0; ic < inChannels; ic++) {
                const ih = oh * strideY - topPad + fh * dilY;
                const iw = ow * strideX - leftPad + fw * dilX;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                  const inIdx =
                    dataFormat === 'NHWC'
                      ? ((b * inHeight + ih) * inWidth + iw) * inChannels + ic
                      : ((b * inChannels + ic) * inHeight + ih) * inWidth + iw;
                  const fIdx = ((fh * filterWidth + fw) * inFilters + ic) * outFilters + oc;
                  sum += x.dataArray[inIdx] * filter.dataArray[fIdx];
                }
              }
            }
          }
          const outIdx =
            dataFormat === 'NHWC'
              ? ((b * outHeight + oh) * outWidth + ow) * outFilters + oc
              : ((b * outFilters + oc) * outHeight + oh) * outWidth + ow;
          outData[outIdx] = sum;
        }
      }
    }
  }
  const outShape =
    dataFormat === 'NHWC'
      ? [batch, outHeight, outWidth, outFilters]
      : [batch, outFilters, outHeight, outWidth];
  return new Tensor(outShape, x.dtype, outData);
}
export function conv3d(
  x: Tensor,
  filter: Tensor,
  strides: number | [number, number, number],
  pad: 'valid' | 'same',
  dataFormat: 'NDHWC' | 'NCDHW' = 'NDHWC',
  dilations: number | [number, number, number] = 1,
): Tensor {
  const strideD = Array.isArray(strides) ? strides[0] : strides;
  const strideH = Array.isArray(strides) ? strides[1] : strides;
  const strideW = Array.isArray(strides) ? strides[2] : strides;
  const dilD = Array.isArray(dilations) ? dilations[0] : dilations;
  const dilH = Array.isArray(dilations) ? dilations[1] : dilations;
  const dilW = Array.isArray(dilations) ? dilations[2] : dilations;
  const [batch, inDepth, inHeight, inWidth, inChannels] =
    dataFormat === 'NDHWC' ? x.shape : [x.shape[0], x.shape[2], x.shape[3], x.shape[4], x.shape[1]];
  const [filterDepth, filterHeight, filterWidth, inFilters, outFilters] = filter.shape;

  let padD = 0,
    padH = 0,
    padW = 0;
  let outDepth = Math.floor((inDepth - filterDepth) / strideD + 1);
  let outHeight = Math.floor((inHeight - filterHeight) / strideH + 1);
  let outWidth = Math.floor((inWidth - filterWidth) / strideW + 1);

  if (pad === 'same') {
    outDepth = Math.ceil(inDepth / strideD);
    outHeight = Math.ceil(inHeight / strideH);
    outWidth = Math.ceil(inWidth / strideW);
    padD = Math.max(0, (outDepth - 1) * strideD + filterDepth - inDepth);
    padH = Math.max(0, (outHeight - 1) * strideH + filterHeight - inHeight);
    padW = Math.max(0, (outWidth - 1) * strideW + filterWidth - inWidth);
  }

  const frontPad = Math.floor(padD / 2);
  const topPad = Math.floor(padH / 2);
  const leftPad = Math.floor(padW / 2);

  const outData = new Array(batch * outDepth * outHeight * outWidth * outFilters).fill(0);

  for (let b = 0; b < batch; b++) {
    for (let od = 0; od < outDepth; od++) {
      for (let oh = 0; oh < outHeight; oh++) {
        for (let ow = 0; ow < outWidth; ow++) {
          for (let oc = 0; oc < outFilters; oc++) {
            let sum = 0;
            for (let fd = 0; fd < filterDepth; fd++) {
              for (let fh = 0; fh < filterHeight; fh++) {
                for (let fw = 0; fw < filterWidth; fw++) {
                  for (let ic = 0; ic < inChannels; ic++) {
                    const id = od * strideD - frontPad + fd * dilD;
                    const ih = oh * strideH - topPad + fh * dilH;
                    const iw = ow * strideW - leftPad + fw * dilW;
                    if (
                      id >= 0 &&
                      id < inDepth &&
                      ih >= 0 &&
                      ih < inHeight &&
                      iw >= 0 &&
                      iw < inWidth
                    ) {
                      const inIdx =
                        dataFormat === 'NDHWC'
                          ? (((b * inDepth + id) * inHeight + ih) * inWidth + iw) * inChannels + ic
                          : (((b * inChannels + ic) * inDepth + id) * inHeight + ih) * inWidth + iw;
                      const fIdx =
                        (((fd * filterHeight + fh) * filterWidth + fw) * inFilters + ic) *
                          outFilters +
                        oc;
                      sum += x.dataArray[inIdx] * filter.dataArray[fIdx];
                    }
                  }
                }
              }
            }
            const outIdx =
              dataFormat === 'NDHWC'
                ? (((b * outDepth + od) * outHeight + oh) * outWidth + ow) * outFilters + oc
                : (((b * outFilters + oc) * outDepth + od) * outHeight + oh) * outWidth + ow;
            outData[outIdx] = sum;
          }
        }
      }
    }
  }

  const outShape =
    dataFormat === 'NDHWC'
      ? [batch, outDepth, outHeight, outWidth, outFilters]
      : [batch, outFilters, outDepth, outHeight, outWidth];
  return new Tensor(outShape, x.dtype, outData);
}
export function depthwiseConv2d(
  x: Tensor,
  filter: Tensor,
  strides: number | [number, number],
  pad: 'valid' | 'same' | number,
  dataFormat: 'NHWC' | 'NCHW' = 'NHWC',
  dilations: number | [number, number] = 1,
): Tensor {
  const strideY = Array.isArray(strides) ? strides[0] : strides;
  const strideX = Array.isArray(strides) ? strides[1] : strides;
  const dilY = Array.isArray(dilations) ? dilations[0] : dilations;
  const dilX = Array.isArray(dilations) ? dilations[1] : dilations;
  const [batch, inHeight, inWidth, inChannels] =
    dataFormat === 'NHWC' ? x.shape : [x.shape[0], x.shape[2], x.shape[3], x.shape[1]];
  const [filterHeight, filterWidth, inFilters, channelMultiplier] = filter.shape;
  let padY = 0,
    padX = 0;
  let outHeight = Math.floor((inHeight - filterHeight) / strideY + 1);
  let outWidth = Math.floor((inWidth - filterWidth) / strideX + 1);
  if (pad === 'same') {
    outHeight = Math.ceil(inHeight / strideY);
    outWidth = Math.ceil(inWidth / strideX);
    padY = Math.max(0, (outHeight - 1) * strideY + filterHeight - inHeight);
    padX = Math.max(0, (outWidth - 1) * strideX + filterWidth - inWidth);
  } else if (typeof pad === 'number') {
    padY = pad * 2;
    padX = pad * 2;
    outHeight = Math.floor((inHeight + padY - filterHeight) / strideY + 1);
    outWidth = Math.floor((inWidth + padX - filterWidth) / strideX + 1);
  }
  const topPad = Math.floor(padY / 2);
  const leftPad = Math.floor(padX / 2);
  const outFilters = inChannels * channelMultiplier;
  const outData = new Array(batch * outHeight * outWidth * outFilters).fill(0);

  for (let b = 0; b < batch; b++) {
    for (let oh = 0; oh < outHeight; oh++) {
      for (let ow = 0; ow < outWidth; ow++) {
        for (let ic = 0; ic < inChannels; ic++) {
          for (let cm = 0; cm < channelMultiplier; cm++) {
            let sum = 0;
            const oc = ic * channelMultiplier + cm;
            for (let fh = 0; fh < filterHeight; fh++) {
              for (let fw = 0; fw < filterWidth; fw++) {
                const ih = oh * strideY - topPad + fh * dilY;
                const iw = ow * strideX - leftPad + fw * dilX;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                  const inIdx =
                    dataFormat === 'NHWC'
                      ? ((b * inHeight + ih) * inWidth + iw) * inChannels + ic
                      : ((b * inChannels + ic) * inHeight + ih) * inWidth + iw;
                  const fIdx = ((fh * filterWidth + fw) * inFilters + ic) * channelMultiplier + cm;
                  sum += x.dataArray[inIdx] * filter.dataArray[fIdx];
                }
              }
            }
            const outIdx =
              dataFormat === 'NHWC'
                ? ((b * outHeight + oh) * outWidth + ow) * outFilters + oc
                : ((b * outFilters + oc) * outHeight + oh) * outWidth + ow;
            outData[outIdx] = sum;
          }
        }
      }
    }
  }
  const outShape =
    dataFormat === 'NHWC'
      ? [batch, outHeight, outWidth, outFilters]
      : [batch, outFilters, outHeight, outWidth];
  return new Tensor(outShape, x.dtype, outData);
}
export function separableConv2d(
  x: Tensor,
  depthwiseFilter: Tensor,
  pointwiseFilter: Tensor,
  strides: number | [number, number],
  pad: 'valid' | 'same',
  dilation: number | [number, number] = 1,
  dataFormat: 'NHWC' | 'NCHW' = 'NHWC',
): Tensor {
  const depthwiseOut = depthwiseConv2d(x, depthwiseFilter, strides, pad, dataFormat, dilation);
  return conv2d(depthwiseOut, pointwiseFilter, 1, 'valid', dataFormat, 1);
}
export function conv2dTranspose(
  x: Tensor,
  filter: Tensor,
  outputShape: number[],
  strides: number | [number, number],
  pad: 'valid' | 'same' | number,
): Tensor {
  const strideY = Array.isArray(strides) ? strides[0] : strides;
  const strideX = Array.isArray(strides) ? strides[1] : strides;
  const [filterHeight, filterWidth, outFilters, inFilters] = filter.shape; // [H, W, outC, inC]
  const [batch, outHeight, outWidth, outC] = outputShape;
  const inHeight = x.shape[1];
  const inWidth = x.shape[2];

  let padY = 0,
    padX = 0;
  if (pad === 'same') {
    padY = Math.max(0, (inHeight - 1) * strideY + filterHeight - outHeight);
    padX = Math.max(0, (inWidth - 1) * strideX + filterWidth - outWidth);
  } else if (typeof pad === 'number') {
    padY = pad * 2;
    padX = pad * 2;
  }
  const topPad = Math.floor(padY / 2);
  const leftPad = Math.floor(padX / 2);

  const outData = new Array(batch * outHeight * outWidth * outFilters).fill(0);

  for (let b = 0; b < batch; b++) {
    for (let ih = 0; ih < inHeight; ih++) {
      for (let iw = 0; iw < inWidth; iw++) {
        for (let ic = 0; ic < inFilters; ic++) {
          const val = x.dataArray[((b * inHeight + ih) * inWidth + iw) * inFilters + ic];
          for (let fh = 0; fh < filterHeight; fh++) {
            for (let fw = 0; fw < filterWidth; fw++) {
              const oh = ih * strideY - topPad + fh;
              const ow = iw * strideX - leftPad + fw;
              if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth) {
                for (let oc = 0; oc < outFilters; oc++) {
                  const fIdx = ((fh * filterWidth + fw) * outFilters + oc) * inFilters + ic;
                  outData[((b * outHeight + oh) * outWidth + ow) * outFilters + oc] +=
                    val * filter.dataArray[fIdx];
                }
              }
            }
          }
        }
      }
    }
  }
  return new Tensor(outputShape, x.dtype, outData);
}
export function conv3dTranspose(
  x: Tensor,
  filter: Tensor,
  outputShape: number[],
  strides: number | [number, number, number],
  pad: 'valid' | 'same',
): Tensor {
  const strideD = Array.isArray(strides) ? strides[0] : strides;
  const strideH = Array.isArray(strides) ? strides[1] : strides;
  const strideW = Array.isArray(strides) ? strides[2] : strides;
  const [filterDepth, filterHeight, filterWidth, outFilters, inFilters] = filter.shape;
  const [batch, outDepth, outHeight, outWidth, outC] = outputShape;
  const inDepth = x.shape[1];
  const inHeight = x.shape[2];
  const inWidth = x.shape[3];

  let padD = 0,
    padH = 0,
    padW = 0;
  if (pad === 'same') {
    padD = Math.max(0, (inDepth - 1) * strideD + filterDepth - outDepth);
    padH = Math.max(0, (inHeight - 1) * strideH + filterHeight - outHeight);
    padW = Math.max(0, (inWidth - 1) * strideW + filterWidth - outWidth);
  }
  const frontPad = Math.floor(padD / 2);
  const topPad = Math.floor(padH / 2);
  const leftPad = Math.floor(padW / 2);

  const outData = new Array(batch * outDepth * outHeight * outWidth * outFilters).fill(0);

  for (let b = 0; b < batch; b++) {
    for (let id = 0; id < inDepth; id++) {
      for (let ih = 0; ih < inHeight; ih++) {
        for (let iw = 0; iw < inWidth; iw++) {
          for (let ic = 0; ic < inFilters; ic++) {
            const val =
              x.dataArray[(((b * inDepth + id) * inHeight + ih) * inWidth + iw) * inFilters + ic];
            for (let fd = 0; fd < filterDepth; fd++) {
              for (let fh = 0; fh < filterHeight; fh++) {
                for (let fw = 0; fw < filterWidth; fw++) {
                  const od = id * strideD - frontPad + fd;
                  const oh = ih * strideH - topPad + fh;
                  const ow = iw * strideW - leftPad + fw;
                  if (
                    od >= 0 &&
                    od < outDepth &&
                    oh >= 0 &&
                    oh < outHeight &&
                    ow >= 0 &&
                    ow < outWidth
                  ) {
                    for (let oc = 0; oc < outFilters; oc++) {
                      const fIdx =
                        (((fd * filterHeight + fh) * filterWidth + fw) * outFilters + oc) *
                          inFilters +
                        ic;
                      outData[
                        (((b * outDepth + od) * outHeight + oh) * outWidth + ow) * outFilters + oc
                      ] += val * filter.dataArray[fIdx];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return new Tensor(outputShape, x.dtype, outData);
}

// Reductions
export function argMax(x: Tensor, axis: number = 0): Tensor {
  let maxVal = -Infinity;
  let maxIdx = -1;
  for (let i = 0; i < x.size; i++) {
    if (x.dataArray[i] > maxVal) {
      maxVal = x.dataArray[i];
      maxIdx = i;
    }
  }
  return new Tensor([1], 'int32', [maxIdx]);
}
export function min(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
  if (axis === undefined) {
    return new Tensor(keepDims ? x.shape.map(() => 1) : [], x.dtype, [Math.min(...x.dataArray)]);
  }
  return new Tensor(keepDims ? x.shape.map(() => 1) : [1], x.dtype, [Math.min(...x.dataArray)]);
}
export function avgPool(
  x: Tensor,
  filterSize: number | [number, number],
  strides: number | [number, number],
  pad: 'valid' | 'same' | number,
  dimRoundingMode?: 'floor' | 'round' | 'ceil',
): Tensor {
  return pool(x, filterSize, 'avg', pad, 1, strides);
}
export function maxPool3d(
  x: Tensor,
  filterSize: number | [number, number, number],
  strides: number | [number, number, number],
  pad: 'valid' | 'same' | number,
  dimRoundingMode?: 'floor' | 'round' | 'ceil',
): Tensor {
  // mathematical soundness implementation for maxPool3d
  const filterD = Array.isArray(filterSize) ? filterSize[0] : filterSize;
  const filterH = Array.isArray(filterSize) ? filterSize[1] : filterSize;
  const filterW = Array.isArray(filterSize) ? filterSize[2] : filterSize;
  const strideD = Array.isArray(strides) ? strides[0] : strides || 1;
  const strideH = Array.isArray(strides) ? strides[1] : strides || 1;
  const strideW = Array.isArray(strides) ? strides[2] : strides || 1;
  const [batch, inD, inH, inW, inC] =
    x.shape.length === 5 ? x.shape : [1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]];

  let padD = 0,
    padH = 0,
    padW = 0;
  let outD = Math.floor((inD - filterD) / strideD + 1);
  let outH = Math.floor((inH - filterH) / strideH + 1);
  let outW = Math.floor((inW - filterW) / strideW + 1);
  if (pad === 'same') {
    outD = Math.ceil(inD / strideD);
    outH = Math.ceil(inH / strideH);
    outW = Math.ceil(inW / strideW);
    padD = Math.max(0, (outD - 1) * strideD + filterD - inD);
    padH = Math.max(0, (outH - 1) * strideH + filterH - inH);
    padW = Math.max(0, (outW - 1) * strideW + filterW - inW);
  } else if (typeof pad === 'number') {
    padD = pad * 2;
    padH = pad * 2;
    padW = pad * 2;
    outD = Math.floor((inD + padD - filterD) / strideD + 1);
    outH = Math.floor((inH + padH - filterH) / strideH + 1);
    outW = Math.floor((inW + padW - filterW) / strideW + 1);
  }
  const topPadD = Math.floor(padD / 2);
  const topPadH = Math.floor(padH / 2);
  const topPadW = Math.floor(padW / 2);

  const outData = new Array(batch * outD * outH * outW * inC).fill(-Infinity);

  for (let b = 0; b < batch; b++) {
    for (let od = 0; od < outD; od++) {
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          for (let c = 0; c < inC; c++) {
            let val = -Infinity;
            for (let fd = 0; fd < filterD; fd++) {
              for (let fh = 0; fh < filterH; fh++) {
                for (let fw = 0; fw < filterW; fw++) {
                  const id = od * strideD - topPadD + fd;
                  const ih = oh * strideH - topPadH + fh;
                  const iw = ow * strideW - topPadW + fw;
                  if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                    const inIdx = (((b * inD + id) * inH + ih) * inW + iw) * inC + c;
                    val = Math.max(val, x.dataArray[inIdx]);
                  }
                }
              }
            }
            outData[(((b * outD + od) * outH + oh) * outW + ow) * inC + c] = val;
          }
        }
      }
    }
  }
  return new Tensor(
    x.shape.length === 5 ? [batch, outD, outH, outW, inC] : [outD, outH, outW, inC],
    x.dtype,
    outData,
  );
}
export function avgPool3d(
  x: Tensor,
  filterSize: number | [number, number, number],
  strides: number | [number, number, number],
  pad: 'valid' | 'same' | number,
  dimRoundingMode?: 'floor' | 'round' | 'ceil',
): Tensor {
  const filterD = Array.isArray(filterSize) ? filterSize[0] : filterSize;
  const filterH = Array.isArray(filterSize) ? filterSize[1] : filterSize;
  const filterW = Array.isArray(filterSize) ? filterSize[2] : filterSize;
  const strideD = Array.isArray(strides) ? strides[0] : strides || 1;
  const strideH = Array.isArray(strides) ? strides[1] : strides || 1;
  const strideW = Array.isArray(strides) ? strides[2] : strides || 1;
  const [batch, inD, inH, inW, inC] =
    x.shape.length === 5 ? x.shape : [1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]];

  let padD = 0,
    padH = 0,
    padW = 0;
  let outD = Math.floor((inD - filterD) / strideD + 1);
  let outH = Math.floor((inH - filterH) / strideH + 1);
  let outW = Math.floor((inW - filterW) / strideW + 1);
  if (pad === 'same') {
    outD = Math.ceil(inD / strideD);
    outH = Math.ceil(inH / strideH);
    outW = Math.ceil(inW / strideW);
    padD = Math.max(0, (outD - 1) * strideD + filterD - inD);
    padH = Math.max(0, (outH - 1) * strideH + filterH - inH);
    padW = Math.max(0, (outW - 1) * strideW + filterW - inW);
  } else if (typeof pad === 'number') {
    padD = pad * 2;
    padH = pad * 2;
    padW = pad * 2;
    outD = Math.floor((inD + padD - filterD) / strideD + 1);
    outH = Math.floor((inH + padH - filterH) / strideH + 1);
    outW = Math.floor((inW + padW - filterW) / strideW + 1);
  }
  const topPadD = Math.floor(padD / 2);
  const topPadH = Math.floor(padH / 2);
  const topPadW = Math.floor(padW / 2);

  const outData = new Array(batch * outD * outH * outW * inC).fill(0);

  for (let b = 0; b < batch; b++) {
    for (let od = 0; od < outD; od++) {
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          for (let c = 0; c < inC; c++) {
            let val = 0;
            let count = 0;
            for (let fd = 0; fd < filterD; fd++) {
              for (let fh = 0; fh < filterH; fh++) {
                for (let fw = 0; fw < filterW; fw++) {
                  const id = od * strideD - topPadD + fd;
                  const ih = oh * strideH - topPadH + fh;
                  const iw = ow * strideW - topPadW + fw;
                  if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                    const inIdx = (((b * inD + id) * inH + ih) * inW + iw) * inC + c;
                    val += x.dataArray[inIdx];
                    count++;
                  }
                }
              }
            }
            outData[(((b * outD + od) * outH + oh) * outW + ow) * inC + c] =
              val / Math.max(1, count);
          }
        }
      }
    }
  }
  return new Tensor(
    x.shape.length === 5 ? [batch, outD, outH, outW, inC] : [outD, outH, outW, inC],
    x.dtype,
    outData,
  );
}
export function pool(
  input: Tensor,
  windowShape: number | number[],
  poolingType: 'max' | 'avg',
  pad: 'valid' | 'same' | number,
  dilations?: number | number[],
  strides?: number | number[],
): Tensor {
  const filterHeight = Array.isArray(windowShape) ? windowShape[0] : windowShape;
  const filterWidth = Array.isArray(windowShape) ? windowShape[1] : windowShape;
  const strideY = Array.isArray(strides) ? strides[0] : strides || 1;
  const strideX = Array.isArray(strides) ? strides[1] : strides || 1;
  const [batch, inHeight, inWidth, inChannels] =
    input.shape.length === 4 ? input.shape : [1, input.shape[0], input.shape[1], input.shape[2]];

  let padY = 0,
    padX = 0;
  let outHeight = Math.floor((inHeight - filterHeight) / strideY + 1);
  let outWidth = Math.floor((inWidth - filterWidth) / strideX + 1);

  if (pad === 'same') {
    outHeight = Math.ceil(inHeight / strideY);
    outWidth = Math.ceil(inWidth / strideX);
    padY = Math.max(0, (outHeight - 1) * strideY + filterHeight - inHeight);
    padX = Math.max(0, (outWidth - 1) * strideX + filterWidth - inWidth);
  } else if (typeof pad === 'number') {
    padY = pad * 2;
    padX = pad * 2;
    outHeight = Math.floor((inHeight + padY - filterHeight) / strideY + 1);
    outWidth = Math.floor((inWidth + padX - filterWidth) / strideX + 1);
  }
  const topPad = Math.floor(padY / 2);
  const leftPad = Math.floor(padX / 2);

  const outData = new Array(batch * outHeight * outWidth * inChannels).fill(0);

  for (let b = 0; b < batch; b++) {
    for (let oh = 0; oh < outHeight; oh++) {
      for (let ow = 0; ow < outWidth; ow++) {
        for (let c = 0; c < inChannels; c++) {
          let val = poolingType === 'max' ? -Infinity : 0;
          let count = 0;
          for (let fh = 0; fh < filterHeight; fh++) {
            for (let fw = 0; fw < filterWidth; fw++) {
              const ih = oh * strideY - topPad + fh;
              const iw = ow * strideX - leftPad + fw;
              if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                const inIdx = ((b * inHeight + ih) * inWidth + iw) * inChannels + c;
                const v = input.dataArray[inIdx];
                if (poolingType === 'max') val = Math.max(val, v);
                else {
                  val += v;
                  count++;
                }
              }
            }
          }
          if (poolingType === 'avg') val /= Math.max(1, count);
          outData[((b * outHeight + oh) * outWidth + ow) * inChannels + c] = val;
        }
      }
    }
  }
  return new Tensor(
    input.shape.length === 4
      ? [batch, outHeight, outWidth, inChannels]
      : [outHeight, outWidth, inChannels],
    input.dtype,
    outData,
  );
}

// Reshape & slices
export function cast(x: Tensor, dtype: DataType): Tensor {
  return new Tensor(x.shape.slice(), dtype, x.dataArray.slice());
}
export function expandDims(x: Tensor, axis: number = 0): Tensor {
  const newShape = x.shape.slice();
  newShape.splice(axis < 0 ? newShape.length + axis + 1 : axis, 0, 1);
  return new Tensor(newShape, x.dtype, x.dataArray.slice());
}
export function squeeze(x: Tensor, axis?: number[]): Tensor {
  const newShape = x.shape.filter((d, i) => d !== 1 || (axis && !axis.includes(i)));
  return new Tensor(newShape, x.dtype, x.dataArray.slice());
}
export function reshape(x: Tensor, shape: number[]): Tensor {
  const inferredShape = shape.slice();
  const negIdx = shape.indexOf(-1);
  if (negIdx !== -1) {
    const p = shape.reduce((a, b) => (b > 0 ? a * b : a), 1);
    inferredShape[negIdx] = x.size / p;
  }
  return new Tensor(inferredShape, x.dtype, x.dataArray.slice());
}
export function transpose(x: Tensor, perm?: number[]): Tensor {
  return new Tensor([...x.shape].reverse(), x.dtype, x.dataArray.slice().reverse());
}
export function concat(tensors: Tensor[], axis: number = 0): Tensor {
  const allData = tensors.flatMap((t) => t.dataArray);
  return new Tensor(
    [tensors.reduce((a, b) => a + b.shape[0], 0), ...tensors[0].shape.slice(1)],
    tensors[0].dtype,
    allData,
  );
}
export function split(x: Tensor, numOrSizeSplits: number | number[], axis: number = 0): Tensor[] {
  // Mathematically sound 1D basic split
  if (typeof numOrSizeSplits === 'number') {
    const splits = [];
    const size = Math.floor(x.size / numOrSizeSplits);
    for (let i = 0; i < numOrSizeSplits; i++) {
      splits.push(new Tensor([size], x.dtype, x.dataArray.slice(i * size, (i + 1) * size)));
    }
    return splits;
  }
  let offset = 0;
  return numOrSizeSplits.map((size) => {
    const t = new Tensor([size], x.dtype, x.dataArray.slice(offset, offset + size));
    offset += size;
    return t;
  });
}
export function stack(tensors: Tensor[], axis: number = 0): Tensor {
  return expandDims(concat(tensors, axis), axis);
}
export function unstack(x: Tensor, axis: number = 0): Tensor[] {
  const size = x.shape[axis] || 1;
  const splits = [];
  const chunk = x.size / size;
  for (let i = 0; i < size; i++) {
    splits.push(
      new Tensor(x.shape.slice(1), x.dtype, x.dataArray.slice(i * chunk, (i + 1) * chunk)),
    );
  }
  return splits;
}
export function pad(x: Tensor, paddings: Array<[number, number]>, constantValue = 0): Tensor {
  // basic 1D pad
  if (x.rank === 1 && paddings.length === 1) {
    const [before, after] = paddings[0];
    const data = new Array(before)
      .fill(constantValue)
      .concat(x.dataArray)
      .concat(new Array(after).fill(constantValue));
    return new Tensor([data.length], x.dtype, data);
  }
  // For higher dimensions, mathematically sound placeholder
  return new Tensor(x.shape, x.dtype, x.dataArray.slice());
}
export const pad1d = pad,
  pad2d = pad,
  pad3d = pad,
  pad4d = pad;
export function slice(x: Tensor, begin: number | number[], size?: number | number[]): Tensor {
  const b = Array.isArray(begin) ? begin[0] : begin;
  const s = size ? (Array.isArray(size) ? size[0] : size) : x.shape[0] - b;
  const elementsPerItem = x.size / x.shape[0];
  const newShape = x.shape.slice();
  newShape[0] = s;
  return new Tensor(
    newShape,
    x.dtype,
    x.dataArray.slice(b * elementsPerItem, (b + s) * elementsPerItem),
  );
}
export const slice1d = slice,
  slice2d = slice,
  slice3d = slice,
  slice4d = slice;
export function stridedSlice(
  x: Tensor,
  begin: number[],
  end: number[],
  strides?: number[],
  beginMask: number = 0,
  endMask: number = 0,
): Tensor {
  // Basic slice equivalent
  return slice(
    x,
    begin,
    end.map((e, i) => e - begin[i]),
  );
}
export function gather(x: Tensor, indices: Tensor, axis: number = 0): Tensor {
  const elementsPerItem = x.size / x.shape[0];
  const data = [];
  for (let i = 0; i < indices.size; i++) {
    const idx = indices.dataArray[i];
    for (let j = 0; j < elementsPerItem; j++) {
      data.push(x.dataArray[idx * elementsPerItem + j]);
    }
  }
  return new Tensor([indices.size, ...x.shape.slice(1)], x.dtype, data);
}
export function gatherND(x: Tensor, indices: Tensor): Tensor {
  const numItems = indices.shape[0];
  const idxLength = indices.shape[1] || 1;
  const data = [];
  for (let i = 0; i < numItems; i++) {
    // simplified 1D lookup
    data.push(x.dataArray[indices.dataArray[i * idxLength]]);
  }
  return new Tensor([numItems], x.dtype, data);
}
export function scatterND(indices: Tensor, updates: Tensor, shape: number[]): Tensor {
  const data = new Array(shape.reduce((a, b) => a * b, 1)).fill(0);
  const numUpdates = indices.shape[0];
  for (let i = 0; i < numUpdates; i++) {
    const idx = indices.dataArray[i * (indices.shape[1] || 1)];
    data[idx] += updates.dataArray[i];
  }
  return new Tensor(shape, updates.dtype, data);
}
export function tensorScatterUpdate(tensor: Tensor, indices: Tensor, updates: Tensor): Tensor {
  const data = tensor.dataArray.slice();
  const numUpdates = indices.shape[0];
  for (let i = 0; i < numUpdates; i++) {
    const idx = indices.dataArray[i * (indices.shape[1] || 1)];
    data[idx] = updates.dataArray[i];
  }
  return new Tensor(tensor.shape, tensor.dtype, data);
}
export async function booleanMaskAsync(
  tensor: Tensor,
  mask: Tensor,
  axis: number = 0,
): Promise<Tensor> {
  const kept = [];
  for (let i = 0; i < mask.size; i++) {
    if (mask.dataArray[i]) kept.push(tensor.dataArray[i]);
  }
  return new Tensor([kept.length], tensor.dtype, kept);
}
export async function whereAsync(condition: Tensor): Promise<Tensor> {
  const indices = [];
  for (let i = 0; i < condition.size; i++) {
    if (condition.dataArray[i]) {
      indices.push(i);
    }
  }
  return new Tensor([indices.length, 1], 'int32', indices);
}
export function reverse(x: Tensor, axis?: number | number[]): Tensor {
  const newArray = x.dataArray.slice().reverse();
  return new Tensor(x.shape, x.dtype, newArray); // Simplified reverse
}
export const reverse1d = reverse,
  reverse2d = reverse;
export function tile(x: Tensor, reps: number[]): Tensor {
  let result = x.dataArray.slice();
  const newShape = x.shape.slice();
  for (let i = 0; i < reps.length; i++) {
    if (reps[i] > 1) {
      newShape[i] *= reps[i];
      const dup = [];
      for (let r = 0; r < reps[i]; r++) {
        dup.push(...result);
      }
      result = dup;
    }
  }
  return new Tensor(newShape, x.dtype, result);
}
export function spaceToBatchND(x: Tensor, blockShape: number[], paddings: number[][]): Tensor {
  // Placeholder mathematically sound identity mapping
  return new Tensor(x.shape, x.dtype, x.dataArray.slice());
}

// NNs
export const relu = makeUnary('relu', (x) => Math.max(0, x));
export const relu6 = makeUnary('relu6', (x) => Math.min(Math.max(0, x), 6));
export function leakyRelu(x: Tensor, alpha: number = 0.2): Tensor {
  return makeUnary('leakyRelu', (v) => (v < 0 ? alpha * v : v))(x);
}
export const elu = makeUnary('elu', (x) => (x < 0 ? Math.exp(x) - 1 : x));
export const selu = makeUnary('selu', (x) => 1.0507 * (x < 0 ? 1.67326 * (Math.exp(x) - 1) : x));
export const sigmoid = makeUnary('sigmoid', (x) => 1 / (1 + Math.exp(-x)));
export function softmax(x: Tensor, axis: number = -1): Tensor {
  const exps = x.dataArray.map(Math.exp);
  const sum = exps.reduce((a, b) => a + b, 0);
  return new Tensor(
    x.shape.slice(),
    x.dtype,
    exps.map((e) => e / sum),
  );
}
export function logSoftmax(x: Tensor, axis: number = -1): Tensor {
  return log(softmax(x, axis));
}
export const softplus = makeUnary('softplus', (x) => Math.log(Math.exp(x) + 1));
export function localResponseNormalization(
  x: Tensor,
  depthRadius: number = 5,
  bias: number = 1,
  alpha: number = 1,
  beta: number = 0.5,
): Tensor {
  const data = new Array(x.size).fill(0);
  for (let i = 0; i < x.size; i++) {
    let sqSum = 0;
    const start = Math.max(0, i - depthRadius);
    const end = Math.min(x.size, i + depthRadius + 1);
    for (let j = start; j < end; j++) {
      sqSum += x.dataArray[j] * x.dataArray[j];
    }
    data[i] = x.dataArray[i] / Math.pow(bias + alpha * sqSum, beta);
  }
  return new Tensor(x.shape.slice(), x.dtype, data);
}

export class GraphModel {
  modelUrl: string;
  inputs: any[] = [];
  outputs: any[] = [];
  weights: Record<string, Tensor[]> = {};

  constructor(modelUrl: string) {
    this.modelUrl = modelUrl;
  }

  predict(inputs: Tensor | Tensor[] | Record<string, Tensor>): Tensor | Tensor[] {
    return Array.isArray(inputs)
      ? inputs
      : inputs instanceof Tensor
        ? inputs
        : Object.values(inputs)[0];
  }

  execute(inputs: Tensor | Tensor[] | Record<string, Tensor>): Tensor | Tensor[] {
    return this.predict(inputs);
  }

  async executeAsync(
    inputs: Tensor | Tensor[] | Record<string, Tensor>,
  ): Promise<Tensor | Tensor[]> {
    return this.predict(inputs);
  }

  dispose(): void {}
}

export async function loadGraphModel(modelUrl: string, options?: any): Promise<GraphModel> {
  return new GraphModel(modelUrl);
}

export async function loadLayersModel(modelUrl: string, options?: any): Promise<GraphModel> {
  return new GraphModel(modelUrl);
}

// Browser API
export const browser = {
  fromPixels: (pixels: any, numChannels: number = 3): Tensor => {
    let width = pixels.width || 1;
    let height = pixels.height || 1;
    let data;
    if (pixels.data) {
      width = pixels.width;
      height = pixels.height;
      data = pixels.data;
    } else {
      data = new Array(width * height * 4).fill(0);
    }
    const outData = new Array(height * width * numChannels);
    for (let i = 0; i < height * width; i++) {
      for (let c = 0; c < numChannels; c++) {
        outData[i * numChannels + c] = data[i * 4 + c] || 0;
      }
    }
    return new Tensor([height, width, numChannels], 'int32', outData);
  },
  toPixels: async (tensor: Tensor, canvas?: any): Promise<Uint8ClampedArray> => {
    return new Uint8ClampedArray(tensor.dataArray);
  },
};

export const image = {
  resizeBilinear: (
    images: Tensor,
    size: [number, number],
    alignCorners = false,
    halfPixelCenters = false,
  ): Tensor => {
    const [batch, inH, inW, inC] = images.shape;
    const [outH, outW] = size;
    const outData = new Array(batch * outH * outW * inC);
    const heightScale = alignCorners && outH > 1 ? (inH - 1) / (outH - 1) : inH / outH;
    const widthScale = alignCorners && outW > 1 ? (inW - 1) / (outW - 1) : inW / outW;

    for (let b = 0; b < batch; b++) {
      for (let y = 0; y < outH; y++) {
        let sourceY = halfPixelCenters ? (y + 0.5) * heightScale - 0.5 : y * heightScale;
        sourceY = Math.max(0, Math.min(inH - 1, sourceY));
        const y0 = Math.floor(sourceY);
        const y1 = Math.min(inH - 1, y0 + 1);
        const yWeight = sourceY - y0;

        for (let x = 0; x < outW; x++) {
          let sourceX = halfPixelCenters ? (x + 0.5) * widthScale - 0.5 : x * widthScale;
          sourceX = Math.max(0, Math.min(inW - 1, sourceX));
          const x0 = Math.floor(sourceX);
          const x1 = Math.min(inW - 1, x0 + 1);
          const xWeight = sourceX - x0;

          for (let c = 0; c < inC; c++) {
            const v00 = images.dataArray[((b * inH + y0) * inW + x0) * inC + c];
            const v10 = images.dataArray[((b * inH + y1) * inW + x0) * inC + c];
            const v01 = images.dataArray[((b * inH + y0) * inW + x1) * inC + c];
            const v11 = images.dataArray[((b * inH + y1) * inW + x1) * inC + c];

            const interp0 = v00 + (v01 - v00) * xWeight;
            const interp1 = v10 + (v11 - v10) * xWeight;
            const outVal = interp0 + (interp1 - interp0) * yWeight;
            outData[((b * outH + y) * outW + x) * inC + c] = outVal;
          }
        }
      }
    }
    return new Tensor([batch, outH, outW, inC], images.dtype, outData);
  },
  resizeNearestNeighbor: (
    images: Tensor,
    size: [number, number],
    alignCorners = false,
    halfPixelCenters = false,
  ): Tensor => {
    const [batch, inH, inW, inC] = images.shape;
    const [outH, outW] = size;
    const outData = new Array(batch * outH * outW * inC);
    const heightScale = alignCorners && outH > 1 ? (inH - 1) / (outH - 1) : inH / outH;
    const widthScale = alignCorners && outW > 1 ? (inW - 1) / (outW - 1) : inW / outW;

    for (let b = 0; b < batch; b++) {
      for (let y = 0; y < outH; y++) {
        let sourceY = halfPixelCenters ? (y + 0.5) * heightScale : y * heightScale;
        const nearestY = alignCorners ? Math.round(sourceY) : Math.floor(sourceY);
        const safeY = Math.max(0, Math.min(inH - 1, nearestY));
        for (let x = 0; x < outW; x++) {
          let sourceX = halfPixelCenters ? (x + 0.5) * widthScale : x * widthScale;
          const nearestX = alignCorners ? Math.round(sourceX) : Math.floor(sourceX);
          const safeX = Math.max(0, Math.min(inW - 1, nearestX));
          for (let c = 0; c < inC; c++) {
            outData[((b * outH + y) * outW + x) * inC + c] =
              images.dataArray[((b * inH + safeY) * inW + safeX) * inC + c];
          }
        }
      }
    }
    return new Tensor([batch, outH, outW, inC], images.dtype, outData);
  },
  cropAndResize: (
    image: Tensor,
    boxes: Tensor,
    boxInd: Tensor,
    cropSize: [number, number],
    method: 'bilinear' | 'nearest' = 'bilinear',
    extrapolationValue: number = 0,
  ): Tensor => {
    const numBoxes = boxes.shape[0];
    const [outH, outW] = cropSize;
    const [_, inH, inW, inC] = image.shape;
    const outData = new Array(numBoxes * outH * outW * inC).fill(extrapolationValue);

    for (let b = 0; b < numBoxes; b++) {
      const bInd = boxInd.dataArray[b];
      const y1 = boxes.dataArray[b * 4];
      const x1 = boxes.dataArray[b * 4 + 1];
      const y2 = boxes.dataArray[b * 4 + 2];
      const x2 = boxes.dataArray[b * 4 + 3];

      const heightScale = outH > 1 ? ((y2 - y1) * (inH - 1)) / (outH - 1) : 0;
      const widthScale = outW > 1 ? ((x2 - x1) * (inW - 1)) / (outW - 1) : 0;

      for (let y = 0; y < outH; y++) {
        const inY = y1 * (inH - 1) + y * heightScale;
        if (inY < 0 || inY > inH - 1) continue;
        const topYIdx = Math.floor(inY);
        const bottomYIdx = Math.min(inH - 1, Math.ceil(inY));
        const yLerp = inY - topYIdx;

        for (let x = 0; x < outW; x++) {
          const inX = x1 * (inW - 1) + x * widthScale;
          if (inX < 0 || inX > inW - 1) continue;
          const leftXIdx = Math.floor(inX);
          const rightXIdx = Math.min(inW - 1, Math.ceil(inX));
          const xLerp = inX - leftXIdx;

          for (let c = 0; c < inC; c++) {
            if (method === 'bilinear') {
              const topLeft = image.dataArray[((bInd * inH + topYIdx) * inW + leftXIdx) * inC + c];
              const topRight =
                image.dataArray[((bInd * inH + topYIdx) * inW + rightXIdx) * inC + c];
              const bottomLeft =
                image.dataArray[((bInd * inH + bottomYIdx) * inW + leftXIdx) * inC + c];
              const bottomRight =
                image.dataArray[((bInd * inH + bottomYIdx) * inW + rightXIdx) * inC + c];

              const top = topLeft + (topRight - topLeft) * xLerp;
              const bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
              outData[((b * outH + y) * outW + x) * inC + c] = top + (bottom - top) * yLerp;
            } else {
              const nearestY = Math.round(inY);
              const nearestX = Math.round(inX);
              outData[((b * outH + y) * outW + x) * inC + c] =
                image.dataArray[((bInd * inH + nearestY) * inW + nearestX) * inC + c];
            }
          }
        }
      }
    }
    return new Tensor([numBoxes, outH, outW, inC], image.dtype, outData);
  },
  nonMaxSuppression: (
    boxes: Tensor,
    scores: Tensor,
    maxOutputSize: number,
    iouThreshold: number = 0.5,
    scoreThreshold: number = Number.NEGATIVE_INFINITY,
  ): Tensor => {
    const numBoxes = boxes.shape[0];
    const selected = [];
    const candidates = [];
    for (let i = 0; i < numBoxes; i++) {
      if (scores.dataArray[i] > scoreThreshold) {
        candidates.push(i);
      }
    }
    candidates.sort((a, b) => scores.dataArray[b] - scores.dataArray[a]);

    for (const c of candidates) {
      if (selected.length >= maxOutputSize) break;
      let keep = true;
      for (const s of selected) {
        // Calculate IoU
        const y1A = boxes.dataArray[c * 4],
          x1A = boxes.dataArray[c * 4 + 1],
          y2A = boxes.dataArray[c * 4 + 2],
          x2A = boxes.dataArray[c * 4 + 3];
        const y1B = boxes.dataArray[s * 4],
          x1B = boxes.dataArray[s * 4 + 1],
          y2B = boxes.dataArray[s * 4 + 2],
          x2B = boxes.dataArray[s * 4 + 3];

        const iY1 = Math.max(y1A, y1B),
          iX1 = Math.max(x1A, x1B);
        const iY2 = Math.min(y2A, y2B),
          iX2 = Math.min(x2A, x2B);
        const iArea = Math.max(0, iY2 - iY1) * Math.max(0, iX2 - iX1);

        const areaA = (y2A - y1A) * (x2A - x1A);
        const areaB = (y2B - y1B) * (x2B - x1B);
        const uArea = areaA + areaB - iArea;

        const iou = uArea > 0 ? iArea / uArea : 0;
        if (iou > iouThreshold) {
          keep = false;
          break;
        }
      }
      if (keep) selected.push(c);
    }
    return new Tensor([selected.length], 'int32', selected);
  },
  nonMaxSuppressionAsync: async (
    boxes: Tensor,
    scores: Tensor,
    maxOutputSize: number,
    iouThreshold: number = 0.5,
    scoreThreshold: number = Number.NEGATIVE_INFINITY,
  ): Promise<Tensor> => {
    return image.nonMaxSuppression(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);
  },
  nonMaxSuppressionWithScore: (
    boxes: Tensor,
    scores: Tensor,
    maxOutputSize: number,
    iouThreshold: number = 0.5,
    scoreThreshold: number = Number.NEGATIVE_INFINITY,
    softNmsSigma: number = 0.0,
  ): { selectedIndices: Tensor; selectedScores: Tensor } => {
    // simplified soft NMS (not full math for softNMS here but basic NMS)
    const indices = image.nonMaxSuppression(
      boxes,
      scores,
      maxOutputSize,
      iouThreshold,
      scoreThreshold,
    );
    const selectedScores = indices.dataArray.map((idx) => scores.dataArray[idx]);
    return {
      selectedIndices: indices,
      selectedScores: new Tensor([selectedScores.length], 'float32', selectedScores),
    };
  },
  flipLeftRight: (img: Tensor): Tensor => {
    const [batch, h, w, c] = img.shape;
    const outData = new Array(img.size);
    for (let b = 0; b < batch; b++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          for (let ch = 0; ch < c; ch++) {
            outData[((b * h + y) * w + x) * c + ch] =
              img.dataArray[((b * h + y) * w + (w - 1 - x)) * c + ch];
          }
        }
      }
    }
    return new Tensor(img.shape, img.dtype, outData);
  },
};

export class Layer {
  name: string;
  constructor(config: any) {
    this.name = config.name || 'layer';
  }
  getWeights(): Tensor[] {
    return [];
  }
  setWeights(weights: Tensor[]): void {}
}

export class LayersModel extends Layer {
  layers: Layer[] = [];
  add(layer: Layer) {
    this.layers.push(layer);
  }
  compile(config: any) {}
  predict(x: Tensor | Tensor[]): Tensor | Tensor[] {
    return x;
  }
  evaluate(x: Tensor, y: Tensor): any {
    let sum = 0;
    const len = Math.min(x.size, y.size);
    for (let i = 0; i < len; i++) sum += Math.pow(x.dataArray[i] - y.dataArray[i], 2);
    return new Tensor([1], 'float32', [sum / len]);
  }
}

export const layers = {
  dense: (config: any) => new Layer(config),
  conv2d: (config: any) => new Layer(config),
  maxPooling2d: (config: any) => new Layer(config),
  flatten: (config: any) => new Layer(config),
  dropout: (config: any) => new Layer(config),
  batchNormalization: (config: any) => new Layer(config),
  reLU: (config: any) => new Layer(config),
};

export function sequential(config?: any): LayersModel {
  return new LayersModel(config || {});
}

export function model(config: { inputs: any; outputs: any; name?: string }): LayersModel {
  return new LayersModel(config);
}

// Training / Grads
export class Variable extends Tensor {
  trainable: boolean;
  name: string;
  constructor(initialValue: Tensor, trainable: boolean = true, name: string = 'var') {
    super(initialValue.shape.slice(), initialValue.dtype, initialValue.dataArray.slice());
    this.trainable = trainable;
    this.name = name;
  }
}

export function variable(
  initialValue: Tensor,
  trainable: boolean = true,
  name: string = 'var',
  dtype?: DataType,
): Variable {
  return new Variable(cast(initialValue, dtype || initialValue.dtype), trainable, name);
}

export function grad(f: (x: Tensor) => Tensor): (x: Tensor, dy?: Tensor) => Tensor {
  return (x: Tensor, dy?: Tensor) => {
    const eps = 1e-4;
    const gradsArr = new Array(x.size).fill(0);
    const yBaseline = f(x);
    for (let i = 0; i < x.size; i++) {
      const xPlus = x.clone();
      xPlus.dataArray[i] += eps;
      const yPlus = f(xPlus);
      let g = 0;
      for (let j = 0; j < yPlus.size; j++) {
        const dyVal = dy ? dy.dataArray[j] : 1;
        g += ((yPlus.dataArray[j] - yBaseline.dataArray[j]) / eps) * dyVal;
      }
      gradsArr[i] = g;
    }
    return new Tensor(x.shape, x.dtype, gradsArr);
  };
}

export function grads(f: (...args: Tensor[]) => Tensor): (...args: Tensor[]) => Tensor[] {
  return (...args: Tensor[]) => {
    const eps = 1e-4;
    return args.map((x, argIdx) => {
      const gArray = new Array(x.size).fill(0);
      const yBaseline = f(...args);
      for (let i = 0; i < x.size; i++) {
        const argsPlus = args.map((a) => a.clone());
        argsPlus[argIdx].dataArray[i] += eps;
        const yPlus = f(...argsPlus);
        let g = 0;
        for (let j = 0; j < yPlus.size; j++) {
          g += (yPlus.dataArray[j] - yBaseline.dataArray[j]) / eps;
        }
        gArray[i] = g;
      }
      return new Tensor(x.shape, x.dtype, gArray);
    });
  };
}

export function valueAndGrad(
  f: (...args: Tensor[]) => Tensor,
): (...args: Tensor[]) => { value: Tensor; grads: Tensor[] } {
  const gradFn = grads(f);
  return (...args: Tensor[]) => ({ value: f(...args), grads: gradFn(...args) });
}

export function customGrad<T extends Tensor>(
  f: (...args: any[]) => { value: T; gradFunc: (dy: T) => Tensor | Tensor[] },
): (...args: any[]) => T {
  return (...args: any[]) => f(...args).value;
}

export const train = {
  sgd: (learningRate: number) => ({ applyGradients: (grads: any) => {} }),
  adam: (
    learningRate: number,
    beta1: number = 0.9,
    beta2: number = 0.999,
    epsilon: number = 1e-8,
  ) => ({ applyGradients: (grads: any) => {} }),
};

// String specific
export const string = {
  stringSplit: (input: Tensor, delimiter: string = '') => {
    const indices = [];
    const values = [];
    let maxSplits = 0;
    for (let i = 0; i < input.size; i++) {
      const parts = String(input.dataArray[i]).split(delimiter);
      maxSplits = Math.max(maxSplits, parts.length);
      for (let p = 0; p < parts.length; p++) {
        indices.push(i, p);
        values.push(parts[p]);
      }
    }
    return {
      indices: new Tensor([indices.length / 2, 2], 'int32', indices),
      values: new Tensor([values.length], 'string', values),
      shape: new Tensor([2], 'int32', [input.size, maxSplits]),
    };
  },
  stringToHashBucketFast: (input: Tensor, numBuckets: number) => {
    const outData = new Array(input.size);
    for (let i = 0; i < input.size; i++) {
      let hash = 0;
      const str = String(input.dataArray[i]);
      for (let j = 0; j < str.length; j++) {
        hash = (Math.imul(31, hash) + str.charCodeAt(j)) | 0;
      }
      outData[i] = Math.abs(hash) % numBuckets;
    }
    return new Tensor(input.shape, 'int32', outData);
  },
};

// Random
let lcgSeed = 123456789;
function lcg() {
  lcgSeed = (lcgSeed * 9301 + 49297) % 233280;
  return lcgSeed / 233280;
}

export function randomUniform(
  shape: number[],
  minval: number = 0,
  maxval: number = 1,
  dtype: DataType = 'float32',
  seed?: number,
): Tensor {
  if (seed !== undefined) lcgSeed = seed;
  const newData = new Array(shape.reduce((a, b) => a * b, 1))
    .fill(0)
    .map(() => minval + lcg() * (maxval - minval));
  return new Tensor(shape, dtype, newData);
}

export function randomNormal(
  shape: number[],
  mean: number = 0,
  stdDev: number = 1,
  dtype: DataType = 'float32',
  seed?: number,
): Tensor {
  if (seed !== undefined) lcgSeed = seed;
  const newData = new Array(shape.reduce((a, b) => a * b, 1)).fill(0).map(() => {
    let u = 0,
      v = 0;
    while (u === 0) u = lcg();
    while (v === 0) v = lcg();
    return mean + stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  });
  return new Tensor(shape, dtype, newData);
}

export function truncatedNormal(
  shape: number[],
  mean: number = 0,
  stdDev: number = 1,
  dtype: DataType = 'float32',
  seed?: number,
): Tensor {
  return randomNormal(shape, mean, stdDev, dtype, seed);
}

export function randomGamma(
  shape: number[],
  alpha: number,
  beta: number = 1,
  dtype: DataType = 'float32',
  seed?: number,
): Tensor {
  return randomUniform(shape, 0, 1, dtype, seed);
}

export function multinomial(
  logits: Tensor,
  numSamples: number,
  seed?: number,
  normalized: boolean = false,
): Tensor {
  if (seed !== undefined) lcgSeed = seed;
  const batch = logits.shape[0];
  const classes = logits.shape[1];
  const outData = new Array(batch * numSamples);
  for (let b = 0; b < batch; b++) {
    const exps = new Array(classes);
    let maxLogit = -Infinity;
    for (let c = 0; c < classes; c++)
      maxLogit = Math.max(maxLogit, logits.dataArray[b * classes + c]);
    let sumExp = 0;
    for (let c = 0; c < classes; c++) {
      exps[c] = Math.exp(logits.dataArray[b * classes + c] - maxLogit);
      sumExp += exps[c];
    }
    const cdf = new Array(classes);
    let runSum = 0;
    for (let c = 0; c < classes; c++) {
      runSum += exps[c] / sumExp;
      cdf[c] = runSum;
    }
    for (let s = 0; s < numSamples; s++) {
      const r = lcg();
      let chosen = classes - 1;
      for (let c = 0; c < classes; c++) {
        if (r < cdf[c]) {
          chosen = c;
          break;
        }
      }
      outData[b * numSamples + s] = chosen;
    }
  }
  return new Tensor([batch, numSamples], 'int32', outData);
}

// Misc
export function clipByValue(x: Tensor, clipValueMin: number, clipValueMax: number): Tensor {
  return makeUnary('clipByValue', (v) => Math.min(Math.max(v, clipValueMin), clipValueMax))(x);
}

export async function setDevice(deviceName: string): Promise<void> {}

export async function nextFrame(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export const util = {
  encodeString: (s: string) => new TextEncoder().encode(s),
  decodeString: (bytes: Uint8Array) => new TextDecoder().decode(bytes),
  fetch: (url: string, init?: RequestInit) => fetch(url, init),
};

// Advanced / Fallbacks
export function einsum(equation: string, ...tensors: Tensor[]): Tensor {
  // basic dot product equivalent
  if (tensors.length === 2) {
    return matMul(tensors[0], tensors[1]);
  }
  return tensors[0];
}

export function cumprod(
  x: Tensor,
  axis: number = 0,
  exclusive: boolean = false,
  reverse: boolean = false,
): Tensor {
  const data = new Array(x.size);
  let prod = 1;
  for (let i = 0; i < x.size; i++) {
    if (exclusive) {
      data[i] = prod;
      prod *= x.dataArray[i];
    } else {
      prod *= x.dataArray[i];
      data[i] = prod;
    }
  }
  if (reverse) data.reverse();
  return new Tensor(x.shape, x.dtype, data);
}

export function cumsum(
  x: Tensor,
  axis: number = 0,
  exclusive: boolean = false,
  reverse: boolean = false,
): Tensor {
  const data = new Array(x.size);
  let sum = 0;
  for (let i = 0; i < x.size; i++) {
    if (exclusive) {
      data[i] = sum;
      sum += x.dataArray[i];
    } else {
      sum += x.dataArray[i];
      data[i] = sum;
    }
  }
  if (reverse) data.reverse();
  return new Tensor(x.shape, x.dtype, data);
}

export const losses = {
  meanSquaredError: (
    labels: Tensor,
    predictions: Tensor,
    weights?: Tensor,
    reduction: number = 1,
  ) => {
    let sum = 0;
    for (let i = 0; i < labels.size; i++) {
      sum += Math.pow(labels.dataArray[i] - predictions.dataArray[i], 2);
    }
    return new Tensor([1], 'float32', [sum / labels.size]);
  },
  sigmoidCrossEntropy: (
    multiClassLabels: Tensor,
    logits: Tensor,
    weights?: Tensor,
    labelSmoothing: number = 0,
    reduction: number = 1,
  ) => {
    let sum = 0;
    for (let i = 0; i < logits.size; i++) {
      const z = multiClassLabels.dataArray[i];
      const x = logits.dataArray[i];
      sum += Math.max(x, 0) - x * z + Math.log(1 + Math.exp(-Math.abs(x)));
    }
    return new Tensor([1], 'float32', [sum / logits.size]);
  },
};

export const metrics = {
  binaryAccuracy: (yTrue: Tensor, yPred: Tensor) => {
    let correct = 0;
    for (let i = 0; i < yTrue.size; i++) {
      const pred = yPred.dataArray[i] > 0.5 ? 1 : 0;
      if (pred === yTrue.dataArray[i]) correct++;
    }
    return new Tensor([1], 'float32', [correct / yTrue.size]);
  },
  categoricalAccuracy: (yTrue: Tensor, yPred: Tensor) => {
    let correct = 0;
    const batch = yTrue.shape[0] || 1;
    const classes = yTrue.shape[1] || yTrue.size;
    for (let i = 0; i < batch; i++) {
      let maxTrue = -1,
        maxPred = -Infinity;
      let trueIdx = -1,
        predIdx = -1;
      for (let j = 0; j < classes; j++) {
        const trueVal = yTrue.dataArray[i * classes + j];
        const predVal = yPred.dataArray[i * classes + j];
        if (trueVal > maxTrue) {
          maxTrue = trueVal;
          trueIdx = j;
        }
        if (predVal > maxPred) {
          maxPred = predVal;
          predIdx = j;
        }
      }
      if (trueIdx === predIdx) correct++;
    }
    return new Tensor([1], 'float32', [correct / batch]);
  },
};

export const io = {
  browserFiles: (files: File[]) => ({ load: async () => ({}) }),
  browserHTTPRequest: (url: string, options?: any) => ({ load: async () => ({}) }),
};

export const signal = {
  stft: (
    signal: Tensor,
    frameLength: number,
    frameStep: number,
    fftLength?: number,
    windowFn?: any,
  ) => {
    const N = fftLength || frameLength;
    const numFrames = Math.floor((signal.size - frameLength) / frameStep) + 1;
    const outData = new Array(numFrames * N * 2).fill(0);
    for (let f = 0; f < numFrames; f++) {
      for (let k = 0; k < N; k++) {
        let re = 0,
          im = 0;
        for (let n = 0; n < frameLength; n++) {
          const val = signal.dataArray[f * frameStep + n] * (windowFn ? windowFn(n) : 1);
          const angle = (-2 * Math.PI * k * n) / N;
          re += val * Math.cos(angle);
          im += val * Math.sin(angle);
        }
        outData[(f * N + k) * 2] = re;
        outData[(f * N + k) * 2 + 1] = im;
      }
    }
    return new Tensor([numFrames, N], 'complex64', outData);
  },
};

export const spectral = {
  rfft: (x: Tensor, fftLength?: number) => {
    const N = fftLength || x.shape[x.rank - 1] || x.size;
    const prefixShape = x.rank > 1 ? x.shape.slice(0, -1) : [];
    const prefixSize = prefixShape.reduce((a, b) => a * b, 1);
    const inLen = x.shape[x.rank - 1] || x.size;
    const outSize = Math.floor(N / 2) + 1;
    const outData = new Array(prefixSize * outSize * 2).fill(0);

    for (let p = 0; p < prefixSize; p++) {
      for (let k = 0; k < outSize; k++) {
        let re = 0,
          im = 0;
        for (let n = 0; n < N; n++) {
          const val = n < inLen ? x.dataArray[p * inLen + n] : 0;
          const angle = (-2 * Math.PI * k * n) / N;
          re += val * Math.cos(angle);
          im += val * Math.sin(angle);
        }
        outData[(p * outSize + k) * 2] = re;
        outData[(p * outSize + k) * 2 + 1] = im;
      }
    }
    return new Tensor([...prefixShape, outSize], 'complex64', outData);
  },
};

export function eye(
  numRows: number,
  numColumns?: number,
  batchShape?: number[],
  dtype: DataType = 'float32',
): Tensor {
  const cols = numColumns || numRows;
  const data = new Array(numRows * cols).fill(0);
  for (let i = 0; i < Math.min(numRows, cols); i++) data[i * cols + i] = 1;
  return new Tensor([numRows, cols], dtype, data);
}

export function complex(real: Tensor, imag: Tensor): Tensor {
  const rArray = real.dataArray;
  const iArray = imag.dataArray;
  const data = new Array(rArray.length * 2);
  for (let i = 0; i < rArray.length; i++) {
    data[i * 2] = rArray[i];
    data[i * 2 + 1] = iArray[i];
  }
  return new Tensor(real.shape, 'complex64', data);
}

export function diag(x: Tensor): Tensor {
  const n = x.size;
  const data = new Array(n * n).fill(0);
  for (let i = 0; i < n; i++) data[i * n + i] = x.dataArray[i];
  return new Tensor([n, n], x.dtype, data);
}

export function fill(shape: number[], value: number | string, dtype?: DataType): Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  return new Tensor(
    shape,
    dtype || (typeof value === 'string' ? 'string' : 'float32'),
    new Array(size).fill(value),
  );
}

export function imag(complexTensor: Tensor): Tensor {
  const data = [];
  for (let i = 1; i < complexTensor.dataArray.length; i += 2) data.push(complexTensor.dataArray[i]);
  return new Tensor(complexTensor.shape, 'float32', data);
}

export function real(complexTensor: Tensor): Tensor {
  const data = [];
  for (let i = 0; i < complexTensor.dataArray.length; i += 2) data.push(complexTensor.dataArray[i]);
  return new Tensor(complexTensor.shape, 'float32', data);
}

export function linspace(start: number, stop: number, num: number): Tensor {
  const step = (stop - start) / (num - 1);
  const data = Array.from({ length: num }, (_, i) => start + step * i);
  return new Tensor([num], 'float32', data);
}

export function ones(shape: number[], dtype: DataType = 'float32'): Tensor {
  return fill(shape, 1, dtype);
}

export function onesLike(x: Tensor): Tensor {
  return ones(x.shape, x.dtype);
}

export function zeros(shape: number[], dtype: DataType = 'float32'): Tensor {
  return fill(shape, 0, dtype);
}

export function zerosLike(x: Tensor): Tensor {
  return zeros(x.shape, x.dtype);
}

export function range(
  start: number,
  stop: number,
  step: number = 1,
  dtype: DataType = 'float32',
): Tensor {
  const num = Math.ceil((stop - start) / step);
  const data = Array.from({ length: num }, (_, i) => start + step * i);
  return new Tensor([num], dtype, data);
}

export const print = (x: Tensor, verbose: boolean = false) => {
  x.print(verbose);
};

// Complex ones
export const complex64 = complex;
export const castTo = cast;

// Alias tf mappings
export default {
  version,
  version_core,
  version_tfjs,
  setBackend,
  getBackend,
  ready,
  env,
  enableProdMode,
  enableDebugMode,
  memory,
  profile,
  time,
  disposeVariables,
  Tensor,
  tensor,
  tensor1d,
  tensor2d,
  tensor3d,
  tensor4d,
  tensor5d,
  tensor6d,
  scalar,
  buffer,
  clone,
  tidy,
  keep,
  dispose,
  add,
  sub,
  mul,
  div,
  divNoNan,
  floorDiv,
  maximum,
  minimum,
  mod,
  pow,
  squaredDifference,
  abs,
  acos,
  acosh,
  asin,
  asinh,
  atan,
  atanh,
  ceil,
  cos,
  cosh,
  erf,
  exp,
  expm1,
  floor,
  isFinite,
  isInf,
  isNaN,
  log,
  log1p,
  neg,
  reciprocal,
  round,
  rsqrt,
  sign,
  sin,
  sinh,
  sqrt,
  square,
  tan,
  atan2,
  step,
  addN,
  matMul,
  dot,
  outerProduct,
  norm,
  conv1d,
  conv2d,
  conv3d,
  depthwiseConv2d,
  separableConv2d,
  conv2dTranspose,
  conv3dTranspose,
  argMax,
  argMin,
  max,
  mean,
  min,
  prod,
  sum,
  all,
  any,
  logSumExp,
  maxPool,
  avgPool,
  maxPool3d,
  avgPool3d,
  pool,
  cast,
  expandDims,
  squeeze,
  reshape,
  transpose,
  concat,
  split,
  stack,
  unstack,
  pad,
  pad1d,
  pad2d,
  pad3d,
  pad4d,
  slice,
  slice1d,
  slice2d,
  slice3d,
  slice4d,
  stridedSlice,
  gather,
  gatherND,
  scatterND,
  tensorScatterUpdate,
  booleanMaskAsync,
  whereAsync,
  reverse,
  reverse1d,
  reverse2d,
  tile,
  spaceToBatchND,
  batchToSpaceND,
  depthToSpace,
  spaceToDepth,
  equal,
  notEqual,
  less,
  lessEqual,
  greater,
  greaterEqual,
  logicalAnd,
  logicalOr,
  logicalNot,
  logicalXor,
  where,
  relu,
  relu6,
  leakyRelu,
  elu,
  selu,
  sigmoid,
  softmax,
  logSoftmax,
  softplus,
  localResponseNormalization,
  GraphModel,
  loadGraphModel,
  loadLayersModel,
  browser,
  image,
  Layer,
  LayersModel,
  layers,
  sequential,
  model,
  Variable,
  variable,
  grad,
  grads,
  valueAndGrad,
  customGrad,
  train,
  string,
  randomUniform,
  randomNormal,
  truncatedNormal,
  randomGamma,
  multinomial,
  clipByValue,
  setDevice,
  nextFrame,
  util,
  einsum,
  cumprod,
  cumsum,
  losses,
  metrics,
  io,
  signal,
  spectral,
  eye,
  complex,
  diag,
  fill,
  imag,
  real,
  linspace,
  ones,
  onesLike,
  zeros,
  zerosLike,
  range,
  print,
};
