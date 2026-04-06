import { describe, it, expect } from 'vitest';
import * as Ops from '../src/ops/index.js';
import { Tensor } from '../src/ir/tensor.js';

describe('Auto-generated Operator Tests', () => {
  it('AbsOp should execute without throwing', () => {
    const op = new Ops.AbsOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AddOp should execute without throwing', () => {
    const op = new Ops.AddOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReluOp should execute without throwing', () => {
    const op = new Ops.ReluOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SubOp should execute without throwing', () => {
    const op = new Ops.SubOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('MulOp should execute without throwing', () => {
    const op = new Ops.MulOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('DivOp should execute without throwing', () => {
    const op = new Ops.DivOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('PowOp should execute without throwing', () => {
    const op = new Ops.PowOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ModOp should execute without throwing', () => {
    const op = new Ops.ModOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('FmodOp should execute without throwing', () => {
    const op = new Ops.FmodOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SignOp should execute without throwing', () => {
    const op = new Ops.SignOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('NegOp should execute without throwing', () => {
    const op = new Ops.NegOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ExpOp should execute without throwing', () => {
    const op = new Ops.ExpOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('LogOp should execute without throwing', () => {
    const op = new Ops.LogOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('Log2Op should execute without throwing', () => {
    const op = new Ops.Log2Op();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('Log10Op should execute without throwing', () => {
    const op = new Ops.Log10Op();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('Expm1Op should execute without throwing', () => {
    const op = new Ops.Expm1Op();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('Log1pOp should execute without throwing', () => {
    const op = new Ops.Log1pOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SinOp should execute without throwing', () => {
    const op = new Ops.SinOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('CosOp should execute without throwing', () => {
    const op = new Ops.CosOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('TanOp should execute without throwing', () => {
    const op = new Ops.TanOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AsinOp should execute without throwing', () => {
    const op = new Ops.AsinOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AcosOp should execute without throwing', () => {
    const op = new Ops.AcosOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AtanOp should execute without throwing', () => {
    const op = new Ops.AtanOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SinhOp should execute without throwing', () => {
    const op = new Ops.SinhOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('CoshOp should execute without throwing', () => {
    const op = new Ops.CoshOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AsinhOp should execute without throwing', () => {
    const op = new Ops.AsinhOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AcoshOp should execute without throwing', () => {
    const op = new Ops.AcoshOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AtanhOp should execute without throwing', () => {
    const op = new Ops.AtanhOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ErfOp should execute without throwing', () => {
    const op = new Ops.ErfOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('IsNaNOp should execute without throwing', () => {
    const op = new Ops.IsNaNOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('IsInfOp should execute without throwing', () => {
    const op = new Ops.IsInfOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('IsFiniteOp should execute without throwing', () => {
    const op = new Ops.IsFiniteOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('BitwiseAndOp should execute without throwing', () => {
    const op = new Ops.BitwiseAndOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('BitwiseOrOp should execute without throwing', () => {
    const op = new Ops.BitwiseOrOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('BitwiseXorOp should execute without throwing', () => {
    const op = new Ops.BitwiseXorOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('BitwiseNotOp should execute without throwing', () => {
    const op = new Ops.BitwiseNotOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('BitShiftOp should execute without throwing', () => {
    const op = new Ops.BitShiftOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('LogicalAndOp should execute without throwing', () => {
    const op = new Ops.LogicalAndOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('LogicalOrOp should execute without throwing', () => {
    const op = new Ops.LogicalOrOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('LogicalXorOp should execute without throwing', () => {
    const op = new Ops.LogicalXorOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('LogicalNotOp should execute without throwing', () => {
    const op = new Ops.LogicalNotOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('EqualOp should execute without throwing', () => {
    const op = new Ops.EqualOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('GreaterOp should execute without throwing', () => {
    const op = new Ops.GreaterOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('GreaterOrEqualOp should execute without throwing', () => {
    const op = new Ops.GreaterOrEqualOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('LessOp should execute without throwing', () => {
    const op = new Ops.LessOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('LessOrEqualOp should execute without throwing', () => {
    const op = new Ops.LessOrEqualOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('MaxOp should execute without throwing', () => {
    const op = new Ops.MaxOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('MinOp should execute without throwing', () => {
    const op = new Ops.MinOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReduceMaxOp should execute without throwing', () => {
    const op = new Ops.ReduceMaxOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReduceMinOp should execute without throwing', () => {
    const op = new Ops.ReduceMinOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReduceSumOp should execute without throwing', () => {
    const op = new Ops.ReduceSumOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReduceMeanOp should execute without throwing', () => {
    const op = new Ops.ReduceMeanOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReduceProdOp should execute without throwing', () => {
    const op = new Ops.ReduceProdOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReduceL1Op should execute without throwing', () => {
    const op = new Ops.ReduceL1Op();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReduceL2Op should execute without throwing', () => {
    const op = new Ops.ReduceL2Op();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReduceLogSumOp should execute without throwing', () => {
    const op = new Ops.ReduceLogSumOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReduceLogSumExpOp should execute without throwing', () => {
    const op = new Ops.ReduceLogSumExpOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReduceSumSquareOp should execute without throwing', () => {
    const op = new Ops.ReduceSumSquareOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ArgMaxOp should execute without throwing', () => {
    const op = new Ops.ArgMaxOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ArgMinOp should execute without throwing', () => {
    const op = new Ops.ArgMinOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('CastOp should execute without throwing', () => {
    const op = new Ops.CastOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('CastLikeOp should execute without throwing', () => {
    const op = new Ops.CastLikeOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReshapeOp should execute without throwing', () => {
    const op = new Ops.ReshapeOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('FlattenOp should execute without throwing', () => {
    const op = new Ops.FlattenOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SqueezeOp should execute without throwing', () => {
    const op = new Ops.SqueezeOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('UnsqueezeOp should execute without throwing', () => {
    const op = new Ops.UnsqueezeOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('TransposeOp should execute without throwing', () => {
    const op = new Ops.TransposeOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ConcatOp should execute without throwing', () => {
    const op = new Ops.ConcatOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SplitOp should execute without throwing', () => {
    const op = new Ops.SplitOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SliceOp should execute without throwing', () => {
    const op = new Ops.SliceOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('GatherOp should execute without throwing', () => {
    const op = new Ops.GatherOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('GatherElementsOp should execute without throwing', () => {
    const op = new Ops.GatherElementsOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('GatherNDOp should execute without throwing', () => {
    const op = new Ops.GatherNDOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ScatterOp should execute without throwing', () => {
    const op = new Ops.ScatterOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ScatterElementsOp should execute without throwing', () => {
    const op = new Ops.ScatterElementsOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ScatterNDOp should execute without throwing', () => {
    const op = new Ops.ScatterNDOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('PadOp should execute without throwing', () => {
    const op = new Ops.PadOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('TileOp should execute without throwing', () => {
    const op = new Ops.TileOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('RepeatOp should execute without throwing', () => {
    const op = new Ops.RepeatOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ExpandOp should execute without throwing', () => {
    const op = new Ops.ExpandOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('WhereOp should execute without throwing', () => {
    const op = new Ops.WhereOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('NonZeroOp should execute without throwing', () => {
    const op = new Ops.NonZeroOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SpaceToDepthOp should execute without throwing', () => {
    const op = new Ops.SpaceToDepthOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('DepthToSpaceOp should execute without throwing', () => {
    const op = new Ops.DepthToSpaceOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('Col2ImOp should execute without throwing', () => {
    const op = new Ops.Col2ImOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('Im2ColOp should execute without throwing', () => {
    const op = new Ops.Im2ColOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('Conv1DOp should execute without throwing', () => {
    const op = new Ops.Conv1DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('Conv2DOp should execute without throwing', () => {
    const op = new Ops.Conv2DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('Conv3DOp should execute without throwing', () => {
    const op = new Ops.Conv3DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ConvTranspose1DOp should execute without throwing', () => {
    const op = new Ops.ConvTranspose1DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ConvTranspose2DOp should execute without throwing', () => {
    const op = new Ops.ConvTranspose2DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ConvTranspose3DOp should execute without throwing', () => {
    const op = new Ops.ConvTranspose3DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('DepthwiseConv2DOp should execute without throwing', () => {
    const op = new Ops.DepthwiseConv2DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('DeformableConv2DOp should execute without throwing', () => {
    const op = new Ops.DeformableConv2DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('MaxPool1DOp should execute without throwing', () => {
    const op = new Ops.MaxPool1DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('MaxPool2DOp should execute without throwing', () => {
    const op = new Ops.MaxPool2DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('MaxPool3DOp should execute without throwing', () => {
    const op = new Ops.MaxPool3DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AveragePool1DOp should execute without throwing', () => {
    const op = new Ops.AveragePool1DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AveragePool2DOp should execute without throwing', () => {
    const op = new Ops.AveragePool2DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AveragePool3DOp should execute without throwing', () => {
    const op = new Ops.AveragePool3DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AdaptiveMaxPool2DOp should execute without throwing', () => {
    const op = new Ops.AdaptiveMaxPool2DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AdaptiveAvgPool2DOp should execute without throwing', () => {
    const op = new Ops.AdaptiveAvgPool2DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('BatchNormOp should execute without throwing', () => {
    const op = new Ops.BatchNormOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('LayerNormOp should execute without throwing', () => {
    const op = new Ops.LayerNormOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('GroupNormOp should execute without throwing', () => {
    const op = new Ops.GroupNormOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('InstanceNormOp should execute without throwing', () => {
    const op = new Ops.InstanceNormOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('LocalResponseNormOp should execute without throwing', () => {
    const op = new Ops.LocalResponseNormOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('RMSNormOp should execute without throwing', () => {
    const op = new Ops.RMSNormOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('AdaLNOp should execute without throwing', () => {
    const op = new Ops.AdaLNOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('LeakyReluOp should execute without throwing', () => {
    const op = new Ops.LeakyReluOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('PReluOp should execute without throwing', () => {
    const op = new Ops.PReluOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('EluOp should execute without throwing', () => {
    const op = new Ops.EluOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('CeluOp should execute without throwing', () => {
    const op = new Ops.CeluOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SeluOp should execute without throwing', () => {
    const op = new Ops.SeluOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SigmoidOp should execute without throwing', () => {
    const op = new Ops.SigmoidOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('HardSigmoidOp should execute without throwing', () => {
    const op = new Ops.HardSigmoidOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('TanhOp should execute without throwing', () => {
    const op = new Ops.TanhOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SoftsignOp should execute without throwing', () => {
    const op = new Ops.SoftsignOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SoftplusOp should execute without throwing', () => {
    const op = new Ops.SoftplusOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('GeluOp should execute without throwing', () => {
    const op = new Ops.GeluOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SiluOp should execute without throwing', () => {
    const op = new Ops.SiluOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('HardSwishOp should execute without throwing', () => {
    const op = new Ops.HardSwishOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('MishOp should execute without throwing', () => {
    const op = new Ops.MishOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SwiGLUOp should execute without throwing', () => {
    const op = new Ops.SwiGLUOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('GeGLUOp should execute without throwing', () => {
    const op = new Ops.GeGLUOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ReGLUOp should execute without throwing', () => {
    const op = new Ops.ReGLUOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('MultiHeadAttentionOp should execute without throwing', () => {
    const op = new Ops.MultiHeadAttentionOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('GroupedQueryAttentionOp should execute without throwing', () => {
    const op = new Ops.GroupedQueryAttentionOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('MultiQueryAttentionOp should execute without throwing', () => {
    const op = new Ops.MultiQueryAttentionOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('FlashAttentionOp should execute without throwing', () => {
    const op = new Ops.FlashAttentionOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('PagedAttentionOp should execute without throwing', () => {
    const op = new Ops.PagedAttentionOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('RoPE1DOp should execute without throwing', () => {
    const op = new Ops.RoPE1DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('RoPE2DOp should execute without throwing', () => {
    const op = new Ops.RoPE2DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('RoPE3DOp should execute without throwing', () => {
    const op = new Ops.RoPE3DOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('ALiBiOp should execute without throwing', () => {
    const op = new Ops.ALiBiOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('SlidingWindowAttentionOp should execute without throwing', () => {
    const op = new Ops.SlidingWindowAttentionOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('StateSpaceModelOp should execute without throwing', () => {
    const op = new Ops.StateSpaceModelOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('RNNOp should execute without throwing', () => {
    const op = new Ops.RNNOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('LSTMOp should execute without throwing', () => {
    const op = new Ops.LSTMOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });

  it('GRUOp should execute without throwing', () => {
    const op = new Ops.GRUOp();
    const d1 = new Float32Array([1, 2]);
    const d2 = new Float32Array([3, 4]);
    const t1 = new Tensor('t1', [2], 'float32', false, true, d1);
    const t2 = new Tensor('t2', [2], 'float32', false, true, d2);

    // Normal execution
    expect(() => op.execute([t1, t2], {})).not.toThrow();

    // Null execution
    expect(() => op.execute([null as Object], {})).not.toThrow();

    // Empty data execution
    const tEmpty = new Tensor('tEmpty', [2], 'float32', false, true, null);
    expect(() => op.execute([tEmpty, tEmpty], {})).not.toThrow();

    // Missing inputs
    expect(() => op.execute([], {})).not.toThrow();

    // One null input for binary
    expect(() => op.execute([t1, null as Object], {})).not.toThrow();
    expect(() => op.execute([null as Object, t2], {})).not.toThrow();
  });
});
