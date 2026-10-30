import { describe, expect, it } from 'vitest';
import type { Tensor } from '../../src/ir/tensor.js';
import {
  type AttributeValue,
  globalRegistry,
  type OpImplementation,
  register_op,
} from '../../src/ops/registry.js';

class MockCustomOp implements OpImplementation {
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    return inputs;
  }
}

describe('OperatorRegistry', () => {
  it('registers, retrieves, falls back, and enumerates operators', () => {
    register_op('ai.onnx', 'MockCustom')(MockCustomOp);
    register_op('', 'MockDefaultDomain', 'webgpu')(MockCustomOp);

    const exact = globalRegistry.get_op('ai.onnx', 'MockCustom');
    expect(exact).toBe(MockCustomOp);

    const fallback = globalRegistry.get_op('ai.onnx', 'MockCustom', 'cpu');
    expect(fallback).toBe(MockCustomOp);

    const missing = globalRegistry.get_op('ai.onnx', 'NonExistentOp', 'cpu');
    expect(missing).toBeUndefined();

    const defaultAll = globalRegistry.getAllRegistered();
    expect(defaultAll['ai.onnx::MockCustom']).toBe(MockCustomOp);

    const webgpuAll = globalRegistry.getAllRegistered('webgpu');
    expect(webgpuAll['ai.onnx::MockDefaultDomain']).toBe(MockCustomOp);
  });
});
