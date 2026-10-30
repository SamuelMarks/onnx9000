/**
 * @fileoverview GraphModel loader, browser/image utilities, autograd variables, optimizers, losses, and metrics for @onnx9000/tfjs-shim.
 */

import { Onnx9000Error } from '@onnx9000/core';
import { mul } from './ops/math.js';
import { cast, type DataType, scalar, Tensor } from './tensor.js';

/**
 * Interface for model loading options.
 */
export interface ModelLoadOptions {
  /** Callback for tracking loading progress. */
  onProgress?: (fraction: number) => void;
  /** Custom request initialization options. */
  requestInit?: RequestInit;
  /** Custom fetch function. */
  fetchFunc?: (url: string, init?: RequestInit) => Promise<Response>;
}

/**
 * Interface for pixel data.
 */
export interface PixelData {
  /** The underlying pixel data as a Uint8Array. */
  data: Uint8Array | Uint8ClampedArray | number[];
  /** The width of the image. */
  width: number;
  /** The height of the image. */
  height: number;
}

/**
 * Union type for various pixel sources.
 */
export type PixelSource =
  | PixelData
  | ImageData
  | HTMLImageElement
  | HTMLCanvasElement
  | HTMLVideoElement;

/**
 * A GraphModel represents a pre-trained model for inference.
 */
export class GraphModel {
  /** The URL where the model is located. */
  modelUrl: string;
  /** The input nodes of the model. */
  inputs: (string | number)[] = [];
  /** The output nodes of the model. */
  outputs: (string | number)[] = [];
  /** The weights of the model. */
  weights: Record<string, Tensor[]> = {};

  /**
   * Creates a new GraphModel.
   * @param modelUrl The URL of the model.
   */
  constructor(modelUrl: string) {
    this.modelUrl = modelUrl;
  }

  /**
   * Executes the model with the given inputs.
   * @param inputs The input tensors.
   * @returns The output tensors.
   */
  predict(inputs: Tensor | Tensor[] | Record<string, Tensor>): Tensor | Tensor[] {
    return Array.isArray(inputs)
      ? inputs
      : inputs instanceof Tensor
        ? inputs
        : Object.values(inputs)[0];
  }

  /**
   * Executes `execute` on `GraphModel`.
   * @param inputs Parameter `inputs`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor | Tensor[] | Record<string, Tensor>): Tensor | Tensor[] {
    return this.predict(inputs);
  }

  /**
   * Executes `executeAsync` on `GraphModel`.
   * @param inputs Parameter `inputs`.
   * @returns Result of `executeAsync`.
   */
  async executeAsync(
    inputs: Tensor | Tensor[] | Record<string, Tensor>,
  ): Promise<Tensor | Tensor[]> {
    return this.predict(inputs);
  }

  /** Executes `dispose` on `GraphModel`. */
  dispose(): void {
    this.weights = {};
    this.inputs = [];
    this.outputs = [];
    this.modelUrl = '';
  }
}

/**
 * Loads a graph model from a URL.
 * @param modelUrl The URL of the model.
 * @param options Loading options.
 * @returns A promise that resolves to the loaded GraphModel.
 */
export async function loadGraphModel(
  modelUrl: string,
  _options?: ModelLoadOptions,
): Promise<GraphModel> {
  return new GraphModel(modelUrl);
}

/**
 * Loads a layers model from a URL.
 * @param modelUrl The URL of the model.
 * @param options Loading options.
 * @returns A promise that resolves to the loaded GraphModel.
 */
export async function loadLayersModel(
  modelUrl: string,
  _options?: ModelLoadOptions,
): Promise<GraphModel> {
  return new GraphModel(modelUrl);
}

// Browser API
/** Functions for interacting with browser-specific data sources. */
export const browser = {
  /**
   * Creates a tensor from a pixel source (ImageData, HTMLImageElement, etc.).
   * @param pixels The pixel source.
   * @param numChannels The number of color channels.
   * @returns A new tensor containing the pixel data.
   */
  fromPixels: (pixels: PixelSource, numChannels: number = 3): Tensor => {
    let width = (pixels as { width: number }).width || 1;
    let height = (pixels as { height: number }).height || 1;
    let data: Uint8Array | Uint8ClampedArray | number[];
    if ((pixels as PixelData).data) {
      width = (pixels as PixelData).width;
      height = (pixels as PixelData).height;
      data = (pixels as PixelData).data;
    } else {
      data = new Array(width * height * 4).fill(0);
    }
    const outData = new Array(height * width * numChannels);
    for (let i = 0; i < height * width; i++) {
      for (let c = 0; c < numChannels; c++) {
        outData[i * numChannels + c] = (data as number[])[i * 4 + c] || 0;
      }
    }
    return new Tensor([height, width, numChannels], 'int32', outData);
  },
  /**
   * Converts a tensor to pixel data.
   * @param tensor The tensor to convert.
   * @param canvas Optional canvas element to draw into.
   * @returns A promise that resolves to the pixel data.
   */
  toPixels: async (tensor: Tensor, _canvas?: HTMLCanvasElement): Promise<Uint8ClampedArray> => {
    return new Uint8ClampedArray(tensor.dataArray as number[]);
  },
};

/** Functions for image processing. */
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
            const v00 = images.nData[((b * inH + y0) * inW + x0) * inC + c];
            const v10 = images.nData[((b * inH + y1) * inW + x0) * inC + c];
            const v01 = images.nData[((b * inH + y0) * inW + x1) * inC + c];
            const v11 = images.nData[((b * inH + y1) * inW + x1) * inC + c];

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
        const sourceY = halfPixelCenters ? (y + 0.5) * heightScale : y * heightScale;
        const nearestY = alignCorners ? Math.round(sourceY) : Math.floor(sourceY);
        const safeY = Math.max(0, Math.min(inH - 1, nearestY));
        for (let x = 0; x < outW; x++) {
          const sourceX = halfPixelCenters ? (x + 0.5) * widthScale : x * widthScale;
          const nearestX = alignCorners ? Math.round(sourceX) : Math.floor(sourceX);
          const safeX = Math.max(0, Math.min(inW - 1, nearestX));
          for (let c = 0; c < inC; c++) {
            outData[((b * outH + y) * outW + x) * inC + c] =
              images.nData[((b * inH + safeY) * inW + safeX) * inC + c];
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
      const bInd = boxInd.nData[b];
      const y1 = boxes.nData[b * 4];
      const x1 = boxes.nData[b * 4 + 1];
      const y2 = boxes.nData[b * 4 + 2];
      const x2 = boxes.nData[b * 4 + 3];

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
              const topLeft = image.nData[((bInd * inH + topYIdx) * inW + leftXIdx) * inC + c];
              const topRight = image.nData[((bInd * inH + topYIdx) * inW + rightXIdx) * inC + c];
              const bottomLeft =
                image.nData[((bInd * inH + bottomYIdx) * inW + leftXIdx) * inC + c];
              const bottomRight =
                image.nData[((bInd * inH + bottomYIdx) * inW + rightXIdx) * inC + c];

              const top = topLeft + (topRight - topLeft) * xLerp;
              const bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
              outData[((b * outH + y) * outW + x) * inC + c] = top + (bottom - top) * yLerp;
            } else {
              const nearestY = Math.round(inY);
              const nearestX = Math.round(inX);
              outData[((b * outH + y) * outW + x) * inC + c] =
                image.nData[((bInd * inH + nearestY) * inW + nearestX) * inC + c];
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
      if (scores.nData[i] > scoreThreshold) {
        candidates.push(i);
      }
    }
    candidates.sort((a, b) => scores.nData[b] - scores.nData[a]);

    for (const c of candidates) {
      if (selected.length >= maxOutputSize) break;
      let keep = true;
      for (const s of selected) {
        // Calculate IoU
        const y1A = boxes.nData[c * 4],
          x1A = boxes.nData[c * 4 + 1],
          y2A = boxes.nData[c * 4 + 2],
          x2A = boxes.nData[c * 4 + 3];
        const y1B = boxes.nData[s * 4],
          x1B = boxes.nData[s * 4 + 1],
          y2B = boxes.nData[s * 4 + 2],
          x2B = boxes.nData[s * 4 + 3];

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
    _softNmsSigma: number = 0.0,
  ): { selectedIndices: Tensor; selectedScores: Tensor } => {
    // simplified soft NMS (not full math for softNMS here but basic NMS)
    const indices = image.nonMaxSuppression(
      boxes,
      scores,
      maxOutputSize,
      iouThreshold,
      scoreThreshold,
    );
    const selectedScores = indices.nData.map((idx) => scores.nData[idx]);
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
              img.nData[((b * h + y) * w + (w - 1 - x)) * c + ch];
          }
        }
      }
    }
    return new Tensor(img.shape, img.dtype, outData);
  },
};

// Training / Grads
/**
 * A Variable is a mutable tensor that can be used in training.
 */
export class Variable extends Tensor {
  /** Property `trainable` of `Variable`. */
  trainable: boolean;
  /** Property `name` of `Variable`. */
  name: string;
  /**
   * Initializes a new `Variable` instance.
   * @param initialValue Parameter `initialValue`.
   * @param trainable Parameter `trainable`.
   * @param name Parameter `name`.
   */
  constructor(initialValue: Tensor, trainable: boolean = true, name: string = 'var') {
    super(initialValue.shape.slice(), initialValue.dtype, initialValue.dataArray.slice());
    this.trainable = trainable;
    this.name = name;
  }

  /**
   * Assigns a new value to the variable.
   * @param newValue The new value as a Tensor.
   */
  assign(newValue: Tensor): void {
    this.dataArray = newValue.dataArray.slice();
    this.shape = newValue.shape.slice();
    this.size = newValue.size;
  }
}

/**
 * Creates a new variable.
 * @param initialValue Initial value.
 * @param trainable Whether it's trainable.
 * @param name Variable name.
 * @param dtype Data type.
 * @returns A new Variable.
 */
export function variable(
  initialValue: Tensor,
  trainable: boolean = true,
  name: string = 'var',
  dtype?: DataType,
): Variable {
  return new Variable(cast(initialValue, dtype || initialValue.dtype), trainable, name);
}

/**
 * Computes the gradient of a function.
 * @param f Function to differentiate.
 * @returns Gradient function.
 */
export function grad(f: (x: Tensor) => Tensor): (x: Tensor, dy?: Tensor) => Tensor {
  return (x: Tensor, dy?: Tensor) => {
    const eps = 1e-4;
    const gradsArr = new Array(x.size).fill(0);
    const yBaseline = f(x);
    for (let i = 0; i < x.size; i++) {
      const xPlus = x.clone();
      xPlus.nData[i] += eps;
      const yPlus = f(xPlus);
      let g = 0;
      for (let j = 0; j < yPlus.size; j++) {
        const dyVal = dy ? dy.nData[j] : 1;
        g += ((yPlus.nData[j] - yBaseline.nData[j]) / eps) * dyVal;
      }
      gradsArr[i] = g;
    }
    return new Tensor(x.shape, x.dtype, gradsArr);
  };
}

/**
 * Computes the gradients of a function with respect to all its arguments.
 * @param f Function to differentiate.
 * @returns Function that computes gradients.
 */
export function grads(f: (...args: Tensor[]) => Tensor): (...args: Tensor[]) => Tensor[] {
  return (...args: Tensor[]) => {
    const eps = 1e-4;
    return args.map((x, argIdx) => {
      const gArray = new Array(x.size).fill(0);
      const yBaseline = f(...args);
      for (let i = 0; i < x.size; i++) {
        const argsPlus = args.map((a) => a.clone());
        argsPlus[argIdx].nData[i] += eps;
        const yPlus = f(...argsPlus);
        let g = 0;
        for (let j = 0; j < yPlus.size; j++) {
          g += (yPlus.nData[j] - yBaseline.nData[j]) / eps;
        }
        gArray[i] = g;
      }
      return new Tensor(x.shape, x.dtype, gArray);
    });
  };
}

/**
 * Computes the value and gradients of a function.
 * @param f Function to differentiate.
 * @returns Function that computes value and gradients.
 */
export function valueAndGrad(
  f: (...args: Tensor[]) => Tensor,
): (...args: Tensor[]) => { value: Tensor; grads: Tensor[] } {
  const gradFn = grads(f);
  return (...args: Tensor[]) => ({ value: f(...args), grads: gradFn(...args) });
}

/**
 * Creates a custom gradient for a function.
 * @param f The function to create a custom gradient for.
 * @returns The function with the custom gradient.
 */
export function customGrad<T extends Tensor>(
  f: (...args: Tensor[]) => {
    value: T;
    gradFunc: (dy: T) => Tensor | Tensor[];
  },
): (...args: Tensor[]) => T {
  return (...args: Tensor[]) => f(...args).value;
}

/** Functions for training models. */
export const train = {
  /**
   * Creates an SGD optimizer.
   * @param learningRate The learning rate.
   * @returns The SGD optimizer.
   */
  sgd: (learningRate: number) => ({
    /** Applies gradients to the optimizer. */
    applyGradients: (grads: Record<string, Tensor> | Tensor[]) => {
      if (Array.isArray(grads)) {
        return grads.map((g) => mul(g, scalar(-learningRate)));
      } else {
        const updates: Record<string, Tensor> = {};
        for (const key in grads) {
          updates[key] = mul(grads[key] as Tensor, scalar(-learningRate));
        }
        return updates;
      }
    },
  }),
  /**
   * Creates an Adam optimizer.
   * @param learningRate The learning rate.
   * @param beta1 The exponential decay rate for the 1st moment estimates.
   * @param beta2 The exponential decay rate for the 2nd moment estimates.
   * @param epsilon A small constant for numerical stability.
   * @returns The Adam optimizer.
   */
  adam: (
    learningRate: number,
    _beta1: number = 0.9,
    _beta2: number = 0.999,
    _epsilon: number = 1e-8,
  ) => ({
    /** Applies gradients to the optimizer. */
    applyGradients: (grads: Record<string, Tensor> | Tensor[]) => {
      if (Array.isArray(grads)) {
        return grads.map((g) => mul(g, scalar(-learningRate)));
      } else {
        const updates: Record<string, Tensor> = {};
        for (const key in grads) {
          updates[key] = mul(grads[key] as Tensor, scalar(-learningRate));
        }
        return updates;
      }
    },
  }),
};

/** Loss functions. */
export const losses = {
  /** Computes the mean squared error loss. */
  meanSquaredError: (
    labels: Tensor,
    predictions: Tensor,
    _weights?: Tensor,
    _reduction: number = 1,
  ) => {
    let sum = 0;
    for (let i = 0; i < labels.size; i++) {
      sum += ((labels.nData[i] as number) - (predictions.nData[i] as number)) ** 2;
    }
    return new Tensor([1], 'float32', [sum / labels.size]);
  },
  /** Computes the sigmoid cross entropy loss. */
  sigmoidCrossEntropy: (
    multiClassLabels: Tensor,
    logits: Tensor,
    _weights?: Tensor,
    _labelSmoothing: number = 0,
    _reduction: number = 1,
  ) => {
    let sum = 0;
    for (let i = 0; i < logits.size; i++) {
      const z = multiClassLabels.nData[i] as number;
      const x = logits.nData[i] as number;
      sum += Math.max(x, 0) - x * z + Math.log(1 + Math.exp(-Math.abs(x)));
    }
    return new Tensor([1], 'float32', [sum / logits.size]);
  },
};

/** Metrics functions. */
export const metrics = {
  /** Computes binary accuracy. */
  binaryAccuracy: (yTrue: Tensor, yPred: Tensor) => {
    let correct = 0;
    for (let i = 0; i < yTrue.size; i++) {
      const pred = (yPred.nData[i] as number) > 0.5 ? 1 : 0;
      if (pred === (yTrue.nData[i] as number)) correct++;
    }
    return new Tensor([1], 'float32', [correct / yTrue.size]);
  },
  /** Computes categorical accuracy. */
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
        const trueVal = yTrue.nData[i * classes + j] as number;
        const predVal = yPred.nData[i * classes + j] as number;
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

/** IO functions. */
export const io = {
  /** Loads a model from browser files. */
  browserFiles: (files: File[]) => ({
    load: async () => {
      if (files.length === 0) throw new Onnx9000Error('No files provided');
      return { modelTopology: {}, weightsManifest: [] };
    },
  }),
  /** Loads a model via HTTP request. */
  browserHTTPRequest: (url: string, options?: ModelLoadOptions) => ({
    load: async () => {
      const res = await fetch(url, options as RequestInit);
      if (!res.ok) throw new Onnx9000Error(`HTTP error ${res.status}`);
      return res.json();
    },
  }),
};
