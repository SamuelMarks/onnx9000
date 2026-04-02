import { describe, it, expect } from 'vitest';
import { globalRegistry, register_op } from '../src/ops/registry.js';
import { AbsOp, AddOp, ReluOp } from '../src/ops/index.js';
import { Tensor } from '../src/ir/tensor.js';

describe('Operator Registry', () => {
  it('should register and retrieve ops', () => {
    const abs = globalRegistry.get_op('ai.onnx', 'Abs');
    expect(abs).toBe(AbsOp);
  });

  it('should handle providers and fallbacks', () => {
    @register_op('test', 'Custom', 'gpu')
    class CustomGpuOp {
      execute() {
        return [];
      }
    }

    @register_op('test', 'Custom')
    class CustomDefaultOp {
      execute() {
        return [];
      }
    }

    expect(globalRegistry.get_op('test', 'Custom', 'gpu')).toBe(CustomGpuOp);
    expect(globalRegistry.get_op('test', 'Custom', 'cpu')).toBe(CustomDefaultOp);
  });

  it('should return all registered ops for a provider', () => {
    const ops = globalRegistry.getAllRegistered(null);
    expect(ops['ai.onnx::Abs']).toBe(AbsOp);
    expect(ops['ai.onnx::Add']).toBe(AddOp);
    expect(ops['ai.onnx::Relu']).toBe(ReluOp);
  });
});

describe('Standard Operators', () => {
  it('AbsOp should execute correctly', () => {
    const op = new AbsOp();
    const data = new Float32Array([-1, 0, 1]);
    const t = new Tensor('t', [3], 'float32', false, true, data);
    const results = op.execute([t], {});
    const outData = results[0].data as Float32Array;
    expect(Array.from(outData)).toEqual([1, 0, 1]);
  });

  it('AddOp should execute correctly', () => {
    const op = new AddOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);
    const results = op.execute([t1, t2], {});
    const outData = results[0].data as Float32Array;
    expect(Array.from(outData)).toEqual([4, 6]);
  });

  it('ReluOp should execute correctly', () => {
    const op = new ReluOp();
    const data = new Float32Array([-1, 0, 1]);
    const t = new Tensor('t', [3], 'float32', false, true, data);
    const results = op.execute([t], {});
    const outData = results[0].data as Float32Array;
    expect(Array.from(outData)).toEqual([0, 0, 1]);
  });

  it('Operators should handle missing data gracefully', () => {
    const t = new Tensor('t', [1], 'float32', false, true, null);
    expect(new AbsOp().execute([t], {})[0]).toBe(t);
    expect(new AddOp().execute([t, t], {})[0]).toBe(t);
    expect(new ReluOp().execute([t], {})[0]).toBe(t);
  });

  it('Operators should handle null or empty inputs', () => {
    expect(new AbsOp().execute([], {})).toEqual([]);
    expect(new ReluOp().execute([], {})).toEqual([]);

    const t = new Tensor('t', [1], 'float32');
    // For AddOp, if both null, returns []
    expect(new AddOp().execute([], {})).toEqual([]);
    // If one null, returns the other one (fallback = a || b)
    expect(new AddOp().execute([t, null as any], {})[0]).toBe(t);
  });
});
