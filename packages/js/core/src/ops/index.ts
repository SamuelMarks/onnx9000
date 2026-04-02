import { register_op, OpImplementation, AttributeValue } from './registry.js';
import { Tensor } from '../ir/tensor.js';

/**
 * Implementation of ONNX Abs operator.
 */
@register_op('ai.onnx', 'Abs')
export class AbsOp implements OpImplementation {
  /**
   * Execute the Abs operation.
   * @param inputs Array of input tensors.
   * @param attributes Dictionary of operator attributes.
   * @returns Array of output tensors.
   */
  execute(inputs: Tensor[], attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x || !x.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (let i = 0; i < view.length; i++) {
      const val = view[i];
      newData[i] = Math.abs(val ?? 0);
    }
    return [new Tensor(x.name + '_abs', x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Add operator.
 */
@register_op('ai.onnx', 'Add')
export class AddOp implements OpImplementation {
  /**
   * Execute the Add operation.
   * @param inputs Array of input tensors.
   * @param attributes Dictionary of operator attributes.
   * @returns Array of output tensors.
   */
  execute(inputs: Tensor[], attributes: Record<string, AttributeValue>): Tensor[] {
    const a = inputs[0];
    const b = inputs[1];
    if (!a || !b || !a.data || !b.data) {
      const fallback = a || b;
      return fallback ? [fallback] : [];
    }
    const viewA =
      a.data instanceof Float32Array
        ? a.data
        : new Float32Array(a.data.buffer, a.data.byteOffset, a.data.byteLength / 4);
    const viewB =
      b.data instanceof Float32Array
        ? b.data
        : new Float32Array(b.data.buffer, b.data.byteOffset, b.data.byteLength / 4);
    const len = Math.max(viewA.length, viewB.length);
    const newData = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      newData[i] = (viewA[i] ?? 0) + (viewB[i] ?? 0);
    }
    return [new Tensor(a.name + '_add', a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Relu operator.
 */
@register_op('ai.onnx', 'Relu')
export class ReluOp implements OpImplementation {
  /**
   * Execute the Relu operation.
   * @param inputs Array of input tensors.
   * @param attributes Dictionary of operator attributes.
   * @returns Array of output tensors.
   */
  execute(inputs: Tensor[], attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x || !x.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (let i = 0; i < view.length; i++) {
      const val = view[i];
      newData[i] = Math.max(0, val ?? 0);
    }
    return [new Tensor(x.name + '_relu', x.shape, x.dtype, false, true, newData)];
  }
}
