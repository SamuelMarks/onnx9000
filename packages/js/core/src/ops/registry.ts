/* eslint-disable */
/**
 * Operator Registry for JS Core.
 */

import { Tensor } from '../ir/tensor.js';

/**
 * Supported attribute value types in ONNX.
 */
export type AttributeValue =
  | number
  | string
  | boolean
  | number[]
  | string[]
  | boolean[]
  | Tensor
  | Tensor[]
  | undefined;

/**
 * Interface for operator implementations.
 */
export interface OpImplementation {
  /**
   * Execute the operator.
   * @param inputs Array of input tensors.
   * @param attributes Dictionary of operator attributes.
   * @returns Array of output tensors.
   */
  execute(inputs: Tensor[], attributes: Record<string, AttributeValue>): Tensor[];
}

/**
 * Registry for mapping ONNX operator types to their implementations.
 */
class OperatorRegistry {
  private registry = new Map<string, new () => OpImplementation>();

  /**
   * Register an operator.
   * @param domain Operator domain.
   * @param opType Operator type name.
   * @param provider Hardware provider name.
   */
  register_op(domain: string, opType: string, provider: string | null = null) {
    const key = this.getKey(domain, opType, provider);
    return (target: new () => OpImplementation) => {
      this.registry.set(key, target);
    };
  }

  /**
   * Get an operator implementation.
   * @param domain Operator domain.
   * @param opType Operator type name.
   * @param provider Hardware provider name.
   */
  get_op(
    domain: string,
    opType: string,
    provider: string | null = null,
  ): (new () => OpImplementation) | undefined {
    const key = this.getKey(domain, opType, provider);
    let impl = this.registry.get(key);
    if (!impl && provider !== null) {
      // Fallback to no provider
      impl = this.registry.get(this.getKey(domain, opType, null));
    }
    return impl;
  }

  private getKey(domain: string, opType: string, provider: string | null): string {
    return `${domain || 'ai.onnx'}::${opType}${provider ? `::${provider}` : ''}`;
  }

  /**
   * Get all registered operators for a provider.
   * @param provider Hardware provider name.
   */
  getAllRegistered(provider: string | null = null): Record<string, new () => OpImplementation> {
    const result: Record<string, new () => OpImplementation> = {};
    for (const [key, value] of this.registry.entries()) {
      const parts = key.split('::');
      const p = parts.length > 2 ? parts[2] : null;
      if (p === (provider || null)) {
        result[`${parts[0]}::${parts[1]}`] = value;
      }
    }
    return result;
  }
}

/**
 * Global operator registry instance.
 */
export const globalRegistry = new OperatorRegistry();

/**
 * Decorator to register an operator.
 * @param domain Operator domain.
 * @param opType Operator type name.
 * @param provider Hardware provider name.
 */
export function register_op(domain: string, opType: string, provider: string | null = null) {
  return globalRegistry.register_op(domain, opType, provider);
}
