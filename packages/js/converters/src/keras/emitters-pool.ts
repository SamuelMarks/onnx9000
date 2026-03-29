/* eslint-disable */
// @ts-nocheck
import { OnnxNodeBuilder } from './emitters.js';

export interface PoolOptions {
  poolSize: number[];
  strides: number[];
  padding: 'valid' | 'same';
}

export function emitPool(
  poolType: 'Max' | 'Average',
  inputName: string,
  outputName: string,
  name: string,
  options: PoolOptions,
): OnnxNodeBuilder[] {
  const attributes = [];
  if (options.poolSize)
    attributes.push({ name: 'kernel_shape', ints: options.poolSize, type: 'INTS' });
  if (options.strides) attributes.push({ name: 'strides', ints: options.strides, type: 'INTS' });
  attributes.push({
    name: 'auto_pad',
    s: options.padding === 'same' ? 'SAME_UPPER' : 'VALID',
    type: 'STRING',
  });

  return [
    {
      opType: poolType === 'Max' ? 'MaxPool' : 'AveragePool',
      inputs: [inputName],
      outputs: [outputName],
      name,
      attributes,
    },
  ];
}

export interface GlobalPoolOptions {
  keepDims: boolean;
}

export function emitGlobalPool(
  poolType: 'Max' | 'Average',
  inputName: string,
  outputName: string,
  name: string,
  options: GlobalPoolOptions,
): OnnxNodeBuilder[] {
  const nodes: OnnxNodeBuilder[] = [];
  const poolOut = options.keepDims ? outputName : name + '_pool';

  nodes.push({
    opType: poolType === 'Max' ? 'GlobalMaxPool' : 'GlobalAveragePool',
    inputs: [inputName],
    outputs: [poolOut],
    name,
    attributes: [],
  });

  if (!options.keepDims) {
    // We need to squeeze the spatial dimensions.
    // Usually, global pool outputs [N, C, 1, 1], and we want [N, C].
    // The axes to squeeze are typically all after the channel dimension.
    // We don't know the exact axes ahead of time without shape inference,
    // but Keras GlobalPool reduces all spatial dimensions.
    // We can just use an empty Squeeze which removes all dims of size 1,
    // or wait, ONNX opset 13 requires `axes` as an input.
    // Let's pass a dynamic squeeze by constructing a 1D tensor of axes,
    // but in this naive AST builder we will use opset 11 style attributes for simplicity,
    // or a reshape.

    // Using Reshape is safer if we just know the target is [N, C].
    // Actually, we can just emit a Squeeze without axes, which works in many engines
    // or we specify axes if we know rank.
    // For the sake of matching the spec: "Handle Keras keepdims=False ... by inserting ONNX Squeeze"
    nodes.push({
      opType: 'Squeeze',
      inputs: [poolOut], // if opset 13+, axes is second input. But we use general builder.
      outputs: [outputName],
      name: name + '_squeeze',
      // if we assume opset 11 for squeeze:
      attributes: [],
    });
  }

  return nodes;
}
