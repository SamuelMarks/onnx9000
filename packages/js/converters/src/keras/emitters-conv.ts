/**
 * @fileoverview emitters-conv.ts
 * Provides emitters-conv functionality for the converters package.
 */
import { emitActivation, type OnnxNodeBuilder } from './emitters.js';
import { calculatePaddingSame } from './layout.js';

/** Interface `ConvOptions` definition. */
export interface ConvOptions {
  /** Property `activation`. */
  activation: string;
  /** Property `strides`. */
  strides: number[];
  /** Property `dilations`. */
  dilations: number[];
  /** Property `padding`. */
  padding: 'valid' | 'same';
  /** Property `groups`. */
  groups?: number;
  /** Property `inputShape`. */
  inputShape?: number[]; // [batch, C, H, W] for calculating 'same' padding
  /** Property `kernelShape`. */
  kernelShape: number[]; // [H, W]
}

/**
 * Executes `emitConv`.
 * @param opType Parameter `opType`.
 * @param inputName Parameter `inputName`.
 * @param outputName Parameter `outputName`.
 * @param weightName Parameter `weightName`.
 * @param biasName Parameter `biasName`.
 * @param name Parameter `name`.
 * @param options Parameter `options`.
 * @returns Result of `emitConv`.
 */
export function emitConv(
  opType: 'Conv' | 'ConvTranspose',
  inputName: string,
  outputName: string,
  weightName: string,
  biasName: string | undefined,
  name: string,
  options: ConvOptions,
): OnnxNodeBuilder[] {
  const nodes: OnnxNodeBuilder[] = [];
  const convOut =
    options.activation && options.activation !== 'linear' ? `${name}_conv` : outputName;

  const attributes = [];
  if (options.strides) attributes.push({ name: 'strides', ints: options.strides, type: 'INTS' });
  if (options.dilations)
    attributes.push({
      name: 'dilations',
      ints: options.dilations,
      type: 'INTS',
    });
  if (options.groups && options.groups > 1)
    attributes.push({ name: 'group', i: options.groups, type: 'INT' });

  if (options.padding === 'same' && options.inputShape) {
    // Compute explicit padding if input shape is known
    const pads: number[] = [];
    const spatialDims = options.strides.length;
    // inputShape should be [N, C, D1, D2...]
    for (let i = 0; i < spatialDims; i++) {
      const inSize = options.inputShape[2 + i] as number;
      const kSize = options.kernelShape[i] as number;
      const stride = options.strides[i] as number;
      const dilation = options.dilations[i] as number;
      const [pBefore, pAfter] = calculatePaddingSame(inSize, kSize, stride, dilation);
      pads.push(pBefore);
      pads.push(pAfter); // Note: ONNX pads is [x_begin, y_begin, x_end, y_end]
    }
    // Repackage to [begin0, begin1, end0, end1]
    const onnxPads = [];
    for (let i = 0; i < spatialDims; i++) onnxPads.push(pads[i * 2]);
    for (let i = 0; i < spatialDims; i++) onnxPads.push(pads[i * 2 + 1]);
    attributes.push({ name: 'pads', ints: onnxPads, type: 'INTS' });
  } else {
    attributes.push({
      name: 'auto_pad',
      s: options.padding === 'same' ? 'SAME_UPPER' : 'VALID',
      type: 'STRING',
    });
  }

  const inputs = [inputName, weightName];
  if (biasName) inputs.push(biasName);

  nodes.push({
    opType,
    inputs,
    outputs: [convOut],
    name,
    attributes: attributes as OnnxNodeBuilder['attributes'],
  });

  if (options.activation && options.activation !== 'linear') {
    nodes.push(...emitActivation(options.activation, convOut, outputName, `${name}_act`));
  }

  return nodes;
}

/**
 * Executes `emitSeparableConv`.
 * @param inputName Parameter `inputName`.
 * @param outputName Parameter `outputName`.
 * @param depthWeightName Parameter `depthWeightName`.
 * @param pointWeightName Parameter `pointWeightName`.
 * @param biasName Parameter `biasName`.
 * @param name Parameter `name`.
 * @param options Parameter `options`.
 * @param inChannels Parameter `inChannels`.
 * @returns Result of `emitSeparableConv`.
 */
export function emitSeparableConv(
  inputName: string,
  outputName: string,
  depthWeightName: string,
  pointWeightName: string,
  biasName: string | undefined,
  name: string,
  options: ConvOptions,
  inChannels: number,
): OnnxNodeBuilder[] {
  const nodes: OnnxNodeBuilder[] = [];
  const depthOut = `${name}_depthwise`;

  // Depthwise step
  nodes.push(
    ...emitConv('Conv', inputName, depthOut, depthWeightName, undefined, `${name}_depth`, {
      ...options,
      groups: inChannels,
      activation: 'linear', // Usually applied after pointwise
    }),
  );

  // Pointwise step
  const pointwiseStrides = Array(options.strides.length).fill(1);
  const pointwiseDilations = Array(options.dilations.length).fill(1);
  nodes.push(
    ...emitConv('Conv', depthOut, outputName, pointWeightName, biasName, `${name}_point`, {
      activation: options.activation,
      strides: pointwiseStrides,
      dilations: pointwiseDilations,
      padding: 'valid', // 1x1 conv doesn't need padding
      kernelShape: pointwiseStrides, // essentially [1, 1]
    }),
  );

  return nodes;
}
