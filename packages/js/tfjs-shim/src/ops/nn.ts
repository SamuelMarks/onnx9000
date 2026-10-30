/**
 * @fileoverview Neural network convolution, pooling, activation, and normalization operations for @onnx9000/tfjs-shim.
 */

import { Tensor } from '../tensor.js';
import { log, makeUnary } from './math.js';

/**
 * Computes 1D convolution.
 * @param x Input tensor.
 * @param filter Filter tensor.
 * @param stride Stride.
 * @param pad Padding type.
 * @param dataFormat Data format.
 * @param dilation Dilation.
 * @returns Resulting tensor.
 */
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
              sum += (x.nData[inIdx] as number) * (filter.nData[fIdx] as number);
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

/**
 * Computes 2D convolution.
 * @param x Input tensor.
 * @param filter Filter tensor.
 * @param strides Strides.
 * @param pad Padding type.
 * @param dataFormat Data format.
 * @param dilations Dilations.
 * @returns Resulting tensor.
 */
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
                  sum += (x.nData[inIdx] as number) * (filter.nData[fIdx] as number);
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

/**
 * Computes 3D convolution.
 * @param x Input tensor.
 * @param filter Filter tensor.
 * @param strides Strides.
 * @param pad Padding type.
 * @param dataFormat Data format.
 * @param dilations Dilations.
 * @returns Resulting tensor.
 */
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
                      sum += (x.nData[inIdx] as number) * (filter.nData[fIdx] as number);
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

/**
 * Computes depthwise 2D convolution.
 * @param x Input tensor.
 * @param filter Filter tensor.
 * @param strides Strides.
 * @param pad Padding type.
 * @param dataFormat Data format.
 * @param dilations Dilations.
 * @returns Resulting tensor.
 */
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
                  sum += (x.nData[inIdx] as number) * (filter.nData[fIdx] as number);
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

/**
 * Computes separable 2D convolution.
 * @param x Input tensor.
 * @param depthwiseFilter Depthwise filter.
 * @param pointwiseFilter Pointwise filter.
 * @param strides Strides.
 * @param pad Padding type.
 * @param dilation Dilation.
 * @param dataFormat Data format.
 * @returns Resulting tensor.
 */
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

/**
 * Computes transposed 2D convolution.
 * @param x Input tensor.
 * @param filter Filter tensor.
 * @param outputShape Output shape.
 * @param strides Strides.
 * @param pad Padding type.
 * @returns Resulting tensor.
 */
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
  const [batch, outHeight, outWidth, _outC] = outputShape;
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
          const val = x.nData[((b * inHeight + ih) * inWidth + iw) * inFilters + ic] as number;
          for (let fh = 0; fh < filterHeight; fh++) {
            for (let fw = 0; fw < filterWidth; fw++) {
              const oh = ih * strideY - topPad + fh;
              const ow = iw * strideX - leftPad + fw;
              if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth) {
                for (let oc = 0; oc < outFilters; oc++) {
                  const fIdx = ((fh * filterWidth + fw) * outFilters + oc) * inFilters + ic;
                  outData[((b * outHeight + oh) * outWidth + ow) * outFilters + oc] +=
                    val * (filter.nData[fIdx] as number);
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

/**
 * Computes transposed 3D convolution.
 * @param x Input tensor.
 * @param filter Filter tensor.
 * @param outputShape Output shape.
 * @param strides Strides.
 * @param pad Padding type.
 * @returns Resulting tensor.
 */
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
  const [batch, outDepth, outHeight, outWidth, _outC] = outputShape;
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
            const val = x.nData[
              (((b * inDepth + id) * inHeight + ih) * inWidth + iw) * inFilters + ic
            ] as number;
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
                      ] += val * (filter.nData[fIdx] as number);
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

/**
 * Performs a 2D max pooling.
 * @param x Input tensor.
 * @param filterSize Size of the pooling window.
 * @param strides Strides of the pooling window.
 * @param pad Padding type.
 * @param dimRoundingMode Rounding mode for output dimensions.
 * @returns Resulting tensor.
 */
export function maxPool(
  x: Tensor,
  filterSize: number | [number, number],
  strides: number | [number, number],
  pad: 'valid' | 'same' | number,
  _dimRoundingMode?: 'floor' | 'round' | 'ceil',
): Tensor {
  return pool(x, filterSize, 'max', pad, 1, strides);
}
/**
 * Performs a 2D average pooling.
 * @param x Input tensor.
 * @param filterSize Size of the pooling window.
 * @param strides Strides of the pooling window.
 * @param pad Padding type.
 * @param dimRoundingMode Rounding mode for output dimensions.
 * @returns Resulting tensor.
 */
export function avgPool(
  x: Tensor,
  filterSize: number | [number, number],
  strides: number | [number, number],
  pad: 'valid' | 'same' | number,
  _dimRoundingMode?: 'floor' | 'round' | 'ceil',
): Tensor {
  return pool(x, filterSize, 'avg', pad, 1, strides);
}
/**
 * Computes the 3D max pooling of the tensor.
 * @param x Input tensor.
 * @param filterSize Filter size.
 * @param strides Strides.
 * @param pad Padding type.
 * @param dataFormat Data format.
 * @param dilations Dilations.
 * @returns Resulting tensor.
 */
export function maxPool3d(
  x: Tensor,
  filterSize: number | [number, number, number],
  strides: number | [number, number, number],
  pad: 'valid' | 'same' | number,
  _dimRoundingMode?: 'floor' | 'round' | 'ceil',
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
                    val = Math.max(val, x.nData[inIdx]);
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
/**
 * Performs a 3D average pooling.
 * @param x Input tensor.
 * @param filterSize Size of the pooling window.
 * @param strides Strides of the pooling window.
 * @param pad Padding type.
 * @param dimRoundingMode Rounding mode for output dimensions.
 * @returns Resulting tensor.
 */
export function avgPool3d(
  x: Tensor,
  filterSize: number | [number, number, number],
  strides: number | [number, number, number],
  pad: 'valid' | 'same' | number,
  _dimRoundingMode?: 'floor' | 'round' | 'ceil',
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
                    val += x.nData[inIdx];
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
/**
 * Performs a pooling operation (max or average) on the input tensor.
 * @param input Input tensor.
 * @param windowShape The shape of the pooling window.
 * @param poolingType The type of pooling to perform ('max' or 'avg').
 * @param pad The type of padding to use.
 * @param dilations Optional dilations for the pooling window.
 * @param strides Optional strides for the pooling window.
 * @returns Resulting tensor.
 */
export function pool(
  input: Tensor,
  windowShape: number | number[],
  poolingType: 'max' | 'avg',
  pad: 'valid' | 'same' | number,
  _dilations?: number | number[],
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
                const v = input.nData[inIdx];
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

// NNs
/** Rectified Linear Unit activation. */
export const relu = makeUnary('relu', (x) => Math.max(0, x));
/** Rectified Linear Unit 6 activation. */
export const relu6 = makeUnary('relu6', (x) => Math.min(Math.max(0, x), 6));

/**
 * Leaky Rectified Linear Unit activation.
 * @param x Input tensor.
 * @param alpha The slope of the negative section.
 * @returns Resulting tensor.
 */
export function leakyRelu(x: Tensor, alpha: number = 0.2): Tensor {
  return makeUnary('leakyRelu', (v) => (v < 0 ? alpha * v : v))(x);
}
/** Exponential Linear Unit activation. */
export const elu = makeUnary('elu', (x) => (x < 0 ? Math.exp(x) - 1 : x));
/** Scaled Exponential Linear Unit activation. */
export const selu = makeUnary('selu', (x) => 1.0507 * (x < 0 ? 1.67326 * (Math.exp(x) - 1) : x));
export const sigmoid = makeUnary('sigmoid', (x) => 1 / (1 + Math.exp(-x)));
/**
 * Computes the softmax activation.
 * @param x Input tensor.
 * @param axis The dimension to compute over.
 * @returns Resulting tensor.
 */
export function softmax(x: Tensor, _axis: number = -1): Tensor {
  const exps = x.nData.map(Math.exp);
  const sum = exps.reduce((a, b) => a + b, 0);
  return new Tensor(
    x.shape.slice(),
    x.dtype,
    exps.map((e) => e / sum),
  );
}
/**
 * Computes the log-softmax activation.
 * @param x Input tensor.
 * @param axis The dimension to compute over.
 * @returns Resulting tensor.
 */
export function logSoftmax(x: Tensor, axis: number = -1): Tensor {
  return log(softmax(x, axis));
}
/** Softplus activation. */
export const softplus = makeUnary('softplus', (x) => Math.log(Math.exp(x) + 1));

/**
 * Local Response Normalization.
 * @param x Input tensor.
 * @param depthRadius Depth radius.
 * @param bias Bias.
 * @param alpha Alpha.
 * @param beta Beta.
 * @returns Resulting tensor.
 */
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
      sqSum += x.nData[j] * x.nData[j];
    }
    data[i] = x.nData[i] / (bias + alpha * sqSum) ** beta;
  }
  return new Tensor(x.shape.slice(), x.dtype, data);
}
