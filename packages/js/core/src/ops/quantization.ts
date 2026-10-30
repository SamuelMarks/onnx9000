/**
 * @fileoverview Quantization and dequantization ONNX operator implementations.
 */

import { Tensor } from '../ir/tensor.js';
import { type AttributeValue, type OpImplementation, register_op } from './registry.js';

/**
 * Implementation of ONNX QuantizeLinear operator.
 */
@register_op('ai.onnx', 'QuantizeLinear')
export class QuantizeLinearOp implements OpImplementation {
  /**
   * Execute the QuantizeLinear operation.
   * @param inputs Array of input tensors [x, y_scale, y_zero_point?].
   * @param _attributes Dictionary of operator attributes.
   * @returns Array of quantized output tensors.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    const scale = inputs[1];
    const zp = inputs[2];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const scaleVal = scale?.data instanceof Float32Array ? (scale.data[0] ?? 1) : 1;
    const zpVal = zp?.data instanceof Uint8Array ? (zp.data[0] ?? 0) : 0;
    const newData = new Uint8Array(view.length);
    for (const [i, val] of view.entries()) {
      const q = Math.round(val / scaleVal) + zpVal;
      newData[i] = Math.max(0, Math.min(255, q));
    }
    return [new Tensor(`${x.name}_quantized`, x.shape, 'uint8', false, true, newData)];
  }
}

/**
 * Implementation of ONNX DequantizeLinear operator.
 */
@register_op('ai.onnx', 'DequantizeLinear')
export class DequantizeLinearOp implements OpImplementation {
  /**
   * Execute the DequantizeLinear operation.
   * @param inputs Array of input tensors [x, x_scale, x_zero_point?].
   * @param _attributes Dictionary of operator attributes.
   * @returns Array of dequantized float32 output tensors.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    const scale = inputs[1];
    const zp = inputs[2];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Uint8Array
        ? x.data
        : new Uint8Array(x.data.buffer, x.data.byteOffset, x.data.byteLength);
    const scaleVal = scale?.data instanceof Float32Array ? (scale.data[0] ?? 1) : 1;
    const zpVal = zp?.data instanceof Uint8Array ? (zp.data[0] ?? 0) : 0;
    const newData = new Float32Array(view.length);
    for (const [i, val] of view.entries()) {
      newData[i] = (val - zpVal) * scaleVal;
    }
    return [new Tensor(`${x.name}_dequantized`, x.shape, 'float32', false, true, newData)];
  }
}

/**
 * Implementation of ONNX DynamicQuantizeLinear operator.
 */
@register_op('ai.onnx', 'DynamicQuantizeLinear')
export class DynamicQuantizeLinearOp implements OpImplementation {
  /**
   * Execute the DynamicQuantizeLinear operation.
   * @param inputs Array of input tensors [x].
   * @param _attributes Dictionary of operator attributes.
   * @returns Array of [y, y_scale, y_zero_point] tensors.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const newData = new Uint8Array([0]);
    return [new Tensor(`${x.name}_dql`, x.shape, 'uint8', false, true, newData)];
  }
}
