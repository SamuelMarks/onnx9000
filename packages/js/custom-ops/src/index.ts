/**
 * @fileoverview Custom operator registry for `@onnx9000/custom-ops`.
 * Provides registration and lookup of user-defined custom operator callbacks.
 */

/**
 * Type signature for a custom operator implementation function.
 */
export type CustomOpFunction = (...args: never[]) => unknown;

/**
 * Registry storing custom operator functions by name.
 */
export class CustomOpRegistry {
  /** Property `ops` of `CustomOpRegistry`. */
  private ops: Map<string, CustomOpFunction>;

  /**
   * Initializes an empty `CustomOpRegistry`.
   */
  constructor() {
    this.ops = new Map();
  }

  /**
   * Registers a custom operator implementation.
   * @param name Operator name.
   * @param func Operator callback function.
   */
  register(name: string, func: CustomOpFunction): void {
    this.ops.set(name, func);
  }

  /**
   * Retrieves a registered custom operator implementation by name.
   * @param name Operator name.
   * @returns The registered function, or `undefined` if not found.
   */
  getOp(name: string): CustomOpFunction | undefined {
    return this.ops.get(name);
  }

  /**
   * Lists all registered custom operator names.
   * @returns Array of operator names.
   */
  listOps(): string[] {
    return Array.from(this.ops.keys());
  }
}

/**
 * Default global `CustomOpRegistry` singleton.
 */
export const registry = new CustomOpRegistry();
