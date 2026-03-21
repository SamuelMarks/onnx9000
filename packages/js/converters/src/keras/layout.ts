/**
 * Translates a generic shape tuple from NHWC (Channels Last) to NCHW (Channels First) format.
 * Applies appropriately to 3D (NLC), 4D (NHWC), and 5D (NDHWC) permutations.
 *
 * @param shape An array of dimension lengths.
 * @returns The permuted array of dimension lengths in NCHW contiguous order.
 */
export function translateNhwcToNchw(shape: number[]): number[] {
  if (shape.length === 4) {
    // [N, H, W, C] -> [N, C, H, W]
    return [shape[0]!, shape[3]!, shape[1]!, shape[2]!];
  } else if (shape.length === 3) {
    // [N, L, C] -> [N, C, L]
    return [shape[0]!, shape[2]!, shape[1]!];
  } else if (shape.length === 5) {
    // [N, D, H, W, C] -> [N, C, D, H, W]
    return [shape[0]!, shape[4]!, shape[1]!, shape[2]!, shape[3]!];
  }
  return [...shape];
}

/**
 * Transpose kernel for 4D Conv2D weights
 * Keras: [H, W, In, Out] -> ONNX: [Out, In, H, W]
 */
export function transposeConv2DWeights(
  data: Float32Array,
  h: number,
  w: number,
  inC: number,
  outC: number,
): Float32Array {
  const out = new Float32Array(data.length);
  for (let o = 0; o < outC; o++) {
    for (let i = 0; i < inC; i++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const srcIdx = y * (w * inC * outC) + x * (inC * outC) + i * outC + o;
          const dstIdx = o * (inC * h * w) + i * (h * w) + y * w + x;
          out[dstIdx] = data[srcIdx]!;
        }
      }
    }
  }
  return out;
}

/**
 * Transpose kernel for 3D Conv1D weights
 * Keras: [L, In, Out] -> ONNX: [Out, In, L]
 */
export function transposeConv1DWeights(
  data: Float32Array,
  l: number,
  inC: number,
  outC: number,
): Float32Array {
  const out = new Float32Array(data.length);
  for (let o = 0; o < outC; o++) {
    for (let i = 0; i < inC; i++) {
      for (let x = 0; x < l; x++) {
        const srcIdx = x * (inC * outC) + i * outC + o;
        const dstIdx = o * (inC * l) + i * l + x;
        out[dstIdx] = data[srcIdx]!;
      }
    }
  }
  return out;
}

/**
 * Transpose kernel for 5D Conv3D weights
 * Keras: [D, H, W, In, Out] -> ONNX: [Out, In, D, H, W]
 */
export function transposeConv3DWeights(
  data: Float32Array,
  d: number,
  h: number,
  w: number,
  inC: number,
  outC: number,
): Float32Array {
  const out = new Float32Array(data.length);
  for (let o = 0; o < outC; o++) {
    for (let i = 0; i < inC; i++) {
      for (let z = 0; z < d; z++) {
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const srcIdx =
              z * (h * w * inC * outC) + y * (w * inC * outC) + x * (inC * outC) + i * outC + o;
            const dstIdx = o * (inC * d * h * w) + i * (d * h * w) + z * (h * w) + y * w + x;
            out[dstIdx] = data[srcIdx]!;
          }
        }
      }
    }
  }
  return out;
}

/**
 * Transpose Keras Dense weights: [In, Out] -> [Out, In]
 */
export function transposeDenseWeights(data: Float32Array, inF: number, outF: number): Float32Array {
  const out = new Float32Array(data.length);
  for (let o = 0; o < outF; o++) {
    for (let i = 0; i < inF; i++) {
      const srcIdx = i * outF + o;
      const dstIdx = o * inF + i;
      out[dstIdx] = data[srcIdx]!;
    }
  }
  return out;
}

export function calculatePaddingSame(
  inputSize: number,
  kernelSize: number,
  stride: number,
  dilation: number = 1,
): [number, number] {
  const effectiveKernelSize = (kernelSize - 1) * dilation + 1;
  let totalPadding = 0;
  if (inputSize % stride === 0) {
    totalPadding = Math.max(effectiveKernelSize - stride, 0);
  } else {
    totalPadding = Math.max(effectiveKernelSize - (inputSize % stride), 0);
  }
  const padBefore = Math.floor(totalPadding / 2);
  const padAfter = totalPadding - padBefore;
  return [padBefore, padAfter];
}

export function calculatePaddingValid(): [number, number] {
  return [0, 0];
}
