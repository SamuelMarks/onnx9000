/**
 * @fileoverview Mathematical, trigonometric, bitwise, and logical ONNX operator implementations.
 */

import { Tensor } from '../ir/tensor.js';
import { type AttributeValue, type OpImplementation, register_op } from './registry.js';

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
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = Math.abs(v);
    }
    return [new Tensor(`${x.name}_abs`, x.shape, x.dtype, false, true, newData)];
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
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
    return [new Tensor(`${a.name}_add`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Sub operator.
 */
@register_op('ai.onnx', 'Sub')
export class SubOp implements OpImplementation {
  /**
   * Executes `execute` on `SubOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => (a ?? 0) - (b ?? 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_sub`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Mul operator.
 */
@register_op('ai.onnx', 'Mul')
export class MulOp implements OpImplementation {
  /**
   * Executes `execute` on `MulOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => (a ?? 0) * (b ?? 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_mul`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Div operator.
 */
@register_op('ai.onnx', 'Div')
export class DivOp implements OpImplementation {
  /**
   * Executes `execute` on `DivOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => (a ?? 0) / (b ?? 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_div`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Pow operator.
 */
@register_op('ai.onnx', 'Pow')
export class PowOp implements OpImplementation {
  /**
   * Executes `execute` on `PowOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => (a ?? 0) ** (b ?? 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_pow`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Mod operator.
 */
@register_op('ai.onnx', 'Mod')
export class ModOp implements OpImplementation {
  /**
   * Executes `execute` on `ModOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => (a ?? 0) % (b ?? 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_mod`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Fmod operator.
 */
@register_op('ai.onnx', 'Fmod')
export class FmodOp implements OpImplementation {
  /**
   * Executes `execute` on `FmodOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => (a ?? 0) % (b ?? 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_fmod`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Sign operator.
 */
@register_op('ai.onnx', 'Sign')
export class SignOp implements OpImplementation {
  /**
   * Executes `execute` on `SignOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.sign(val))(v);
    }
    return [new Tensor(`${x.name}_sign`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Neg operator.
 */
@register_op('ai.onnx', 'Neg')
export class NegOp implements OpImplementation {
  /**
   * Executes `execute` on `NegOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => -val)(v);
    }
    return [new Tensor(`${x.name}_neg`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Exp operator.
 */
@register_op('ai.onnx', 'Exp')
export class ExpOp implements OpImplementation {
  /**
   * Executes `execute` on `ExpOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.exp(val))(v);
    }
    return [new Tensor(`${x.name}_exp`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Log operator.
 */
@register_op('ai.onnx', 'Log')
export class LogOp implements OpImplementation {
  /**
   * Executes `execute` on `LogOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.log(val))(v);
    }
    return [new Tensor(`${x.name}_log`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Log2 operator.
 */
@register_op('ai.onnx', 'Log2')
export class Log2Op implements OpImplementation {
  /**
   * Executes `execute` on `Log2Op`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.log2(val))(v);
    }
    return [new Tensor(`${x.name}_log2`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Log10 operator.
 */
@register_op('ai.onnx', 'Log10')
export class Log10Op implements OpImplementation {
  /**
   * Executes `execute` on `Log10Op`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.log10(val))(v);
    }
    return [new Tensor(`${x.name}_log10`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Expm1 operator.
 */
@register_op('ai.onnx', 'Expm1')
export class Expm1Op implements OpImplementation {
  /**
   * Executes `execute` on `Expm1Op`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.expm1(val))(v);
    }
    return [new Tensor(`${x.name}_expm1`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Log1p operator.
 */
@register_op('ai.onnx', 'Log1p')
export class Log1pOp implements OpImplementation {
  /**
   * Executes `execute` on `Log1pOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.log1p(val))(v);
    }
    return [new Tensor(`${x.name}_log1p`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Sin operator.
 */
@register_op('ai.onnx', 'Sin')
export class SinOp implements OpImplementation {
  /**
   * Executes `execute` on `SinOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.sin(val))(v);
    }
    return [new Tensor(`${x.name}_sin`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Cos operator.
 */
@register_op('ai.onnx', 'Cos')
export class CosOp implements OpImplementation {
  /**
   * Executes `execute` on `CosOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.cos(val))(v);
    }
    return [new Tensor(`${x.name}_cos`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Tan operator.
 */
@register_op('ai.onnx', 'Tan')
export class TanOp implements OpImplementation {
  /**
   * Executes `execute` on `TanOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.tan(val))(v);
    }
    return [new Tensor(`${x.name}_tan`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Asin operator.
 */
@register_op('ai.onnx', 'Asin')
export class AsinOp implements OpImplementation {
  /**
   * Executes `execute` on `AsinOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.asin(val))(v);
    }
    return [new Tensor(`${x.name}_asin`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Acos operator.
 */
@register_op('ai.onnx', 'Acos')
export class AcosOp implements OpImplementation {
  /**
   * Executes `execute` on `AcosOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.acos(val))(v);
    }
    return [new Tensor(`${x.name}_acos`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Atan operator.
 */
@register_op('ai.onnx', 'Atan')
export class AtanOp implements OpImplementation {
  /**
   * Executes `execute` on `AtanOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.atan(val))(v);
    }
    return [new Tensor(`${x.name}_atan`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Sinh operator.
 */
@register_op('ai.onnx', 'Sinh')
export class SinhOp implements OpImplementation {
  /**
   * Executes `execute` on `SinhOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.sinh(val))(v);
    }
    return [new Tensor(`${x.name}_sinh`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Cosh operator.
 */
@register_op('ai.onnx', 'Cosh')
export class CoshOp implements OpImplementation {
  /**
   * Executes `execute` on `CoshOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.cosh(val))(v);
    }
    return [new Tensor(`${x.name}_cosh`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Asinh operator.
 */
@register_op('ai.onnx', 'Asinh')
export class AsinhOp implements OpImplementation {
  /**
   * Executes `execute` on `AsinhOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.asinh(val))(v);
    }
    return [new Tensor(`${x.name}_asinh`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Acosh operator.
 */
@register_op('ai.onnx', 'Acosh')
export class AcoshOp implements OpImplementation {
  /**
   * Executes `execute` on `AcoshOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.acosh(val))(v);
    }
    return [new Tensor(`${x.name}_acosh`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Atanh operator.
 */
@register_op('ai.onnx', 'Atanh')
export class AtanhOp implements OpImplementation {
  /**
   * Executes `execute` on `AtanhOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => Math.atanh(val))(v);
    }
    return [new Tensor(`${x.name}_atanh`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Erf operator.
 */
@register_op('ai.onnx', 'Erf')
export class ErfOp implements OpImplementation {
  /**
   * Executes `execute` on `ErfOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => {
        const x = val;
        const a1 = 0.254829592;
        const a2 = -0.284496736;
        const a3 = 1.421413741;
        const a4 = -1.453152027;
        const a5 = 1.061405429;
        const p = 0.3275911;
        const sign = x < 0 ? -1 : 1;
        const t = 1.0 / (1.0 + p * Math.abs(x));
        const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        return sign * y;
      })(v);
    }
    return [new Tensor(`${x.name}_erf`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX IsNaN operator.
 */
@register_op('ai.onnx', 'IsNaN')
export class IsNaNOp implements OpImplementation {
  /**
   * Executes `execute` on `IsNaNOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => (Number.isNaN(val) ? 1 : 0))(v);
    }
    return [new Tensor(`${x.name}_isnan`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX IsInf operator.
 */
@register_op('ai.onnx', 'IsInf')
export class IsInfOp implements OpImplementation {
  /**
   * Executes `execute` on `IsInfOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => (!Number.isFinite(val) && !Number.isNaN(val) ? 1 : 0))(v);
    }
    return [new Tensor(`${x.name}_isinf`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX IsFinite operator.
 */
@register_op('ai.onnx', 'IsFinite')
export class IsFiniteOp implements OpImplementation {
  /**
   * Executes `execute` on `IsFiniteOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => (Number.isFinite(val) ? 1 : 0))(v);
    }
    return [new Tensor(`${x.name}_isfinite`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX BitwiseAnd operator.
 */
@register_op('ai.onnx', 'BitwiseAnd')
export class BitwiseAndOp implements OpImplementation {
  /**
   * Executes `execute` on `BitwiseAndOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => (a ?? 0) & (b ?? 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_bitwiseand`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX BitwiseOr operator.
 */
@register_op('ai.onnx', 'BitwiseOr')
export class BitwiseOrOp implements OpImplementation {
  /**
   * Executes `execute` on `BitwiseOrOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => (a ?? 0) | (b ?? 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_bitwiseor`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX BitwiseXor operator.
 */
@register_op('ai.onnx', 'BitwiseXor')
export class BitwiseXorOp implements OpImplementation {
  /**
   * Executes `execute` on `BitwiseXorOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => (a ?? 0) ^ (b ?? 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_bitwisexor`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX BitwiseNot operator.
 */
@register_op('ai.onnx', 'BitwiseNot')
export class BitwiseNotOp implements OpImplementation {
  /**
   * Executes `execute` on `BitwiseNotOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => ~val)(v);
    }
    return [new Tensor(`${x.name}_bitwisenot`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX BitShift operator.
 */
@register_op('ai.onnx', 'BitShift')
export class BitShiftOp implements OpImplementation {
  /**
   * Executes `execute` on `BitShiftOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => (a ?? 0) << (b ?? 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_bitshift`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX LogicalAnd operator.
 */
@register_op('ai.onnx', 'LogicalAnd')
export class LogicalAndOp implements OpImplementation {
  /**
   * Executes `execute` on `LogicalAndOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => ((a ?? 0) && (b ?? 0) ? 1 : 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_logicaland`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX LogicalOr operator.
 */
@register_op('ai.onnx', 'LogicalOr')
export class LogicalOrOp implements OpImplementation {
  /**
   * Executes `execute` on `LogicalOrOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => ((a ?? 0) || (b ?? 0) ? 1 : 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_logicalor`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX LogicalXor operator.
 */
@register_op('ai.onnx', 'LogicalXor')
export class LogicalXorOp implements OpImplementation {
  /**
   * Executes `execute` on `LogicalXorOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => ((a ?? 0) ? 1 : 0) ^ ((b ?? 0) ? 1 : 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_logicalxor`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX LogicalNot operator.
 */
@register_op('ai.onnx', 'LogicalNot')
export class LogicalNotOp implements OpImplementation {
  /**
   * Executes `execute` on `LogicalNotOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const x = inputs[0];
    if (!x?.data) {
      return x ? [x] : [];
    }
    const view =
      x.data instanceof Float32Array
        ? x.data
        : new Float32Array(x.data.buffer, x.data.byteOffset, x.data.byteLength / 4);
    const newData = new Float32Array(view.length);
    for (const [i, v] of view.entries()) {
      newData[i] = ((val) => (val ? 0 : 1))(v);
    }
    return [new Tensor(`${x.name}_logicalnot`, x.shape, x.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Equal operator.
 */
@register_op('ai.onnx', 'Equal')
export class EqualOp implements OpImplementation {
  /**
   * Executes `execute` on `EqualOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => ((a ?? 0) === (b ?? 0) ? 1 : 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_equal`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Greater operator.
 */
@register_op('ai.onnx', 'Greater')
export class GreaterOp implements OpImplementation {
  /**
   * Executes `execute` on `GreaterOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => ((a ?? 0) > (b ?? 0) ? 1 : 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_greater`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX GreaterOrEqual operator.
 */
@register_op('ai.onnx', 'GreaterOrEqual')
export class GreaterOrEqualOp implements OpImplementation {
  /**
   * Executes `execute` on `GreaterOrEqualOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => ((a ?? 0) >= (b ?? 0) ? 1 : 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_greaterorequal`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Less operator.
 */
@register_op('ai.onnx', 'Less')
export class LessOp implements OpImplementation {
  /**
   * Executes `execute` on `LessOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => ((a ?? 0) < (b ?? 0) ? 1 : 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_less`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX LessOrEqual operator.
 */
@register_op('ai.onnx', 'LessOrEqual')
export class LessOrEqualOp implements OpImplementation {
  /**
   * Executes `execute` on `LessOrEqualOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => ((a ?? 0) <= (b ?? 0) ? 1 : 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_lessorequal`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Max operator.
 */
@register_op('ai.onnx', 'Max')
export class MaxOp implements OpImplementation {
  /**
   * Executes `execute` on `MaxOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => Math.max(a ?? 0, b ?? 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_max`, a.shape, a.dtype, false, true, newData)];
  }
}

/**
 * Implementation of ONNX Min operator.
 */
@register_op('ai.onnx', 'Min')
export class MinOp implements OpImplementation {
  /**
   * Executes `execute` on `MinOp`.
   * @param inputs Parameter `inputs`.
   * @param _attributes Parameter `_attributes`.
   * @returns Result of `execute`.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
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
      newData[i] = ((a, b) => Math.min(a ?? 0, b ?? 0))(viewA[i], viewB[i]);
    }
    return [new Tensor(`${a.name}_min`, a.shape, a.dtype, false, true, newData)];
  }
}
