import { describe, expect, it } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import * as Ops from '../src/ops/index.js';
import type { OpImplementation } from '../src/ops/registry.js';

describe('Operator Edge Cases and Buffer Views', () => {
  const f32A = new Tensor(
    'a',
    [5],
    'float32',
    false,
    true,
    new Float32Array([0.0, 1.5, -2.5, Number.NaN, Number.POSITIVE_INFINITY]),
  );
  const f32B = new Tensor('b', [2], 'float32', false, true, new Float32Array([0.0, 0.5]));
  const f32C = new Tensor('c', [3], 'float32', false, true, new Float32Array([0.0, 0.5, 2.0]));
  const u8A = new Tensor(
    'u8a',
    [8],
    'uint8',
    false,
    true,
    new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]),
  );
  const u8B = new Tensor(
    'u8b',
    [8],
    'uint8',
    false,
    true,
    new Uint8Array([2, 2, 2, 2, 1, 1, 1, 1]),
  );
  const emptyData = new Tensor('empty', [2], 'float32', false, true, null);
  const emptyScale = new Tensor('emptyScale', [0], 'float32', false, true, new Float32Array([]));
  const emptyZp = new Tensor('emptyZp', [0], 'uint8', false, true, new Uint8Array([]));

  const opClasses = Object.entries(Ops).filter(
    (entry): entry is [string, new () => OpImplementation] =>
      typeof entry[1] === 'function' && entry[0].endsWith('Op'),
  );

  for (const [name, OpClass] of opClasses) {
    it(`${name} handles valid, empty, and partial inputs`, () => {
      const op = new OpClass();
      expect(op.execute([f32A, f32B, u8A], { direction: 'LEFT' }).length).toBeGreaterThanOrEqual(1);
      expect(op.execute([f32B, f32C, f32A], { direction: 'RIGHT' }).length).toBeGreaterThanOrEqual(
        1,
      );
      expect(op.execute([u8A, u8B, u8A], {}).length).toBeGreaterThanOrEqual(1);
      expect(op.execute([emptyData, f32B], {}).length).toBe(1);
      expect(op.execute([f32A, emptyData], {}).length).toBe(1);
      expect(op.execute([emptyData, emptyData], {}).length).toBe(1);
      expect(op.execute([f32A, emptyScale, emptyZp], {}).length).toBeGreaterThanOrEqual(1);
      expect(op.execute([u8A, emptyScale, emptyZp], {}).length).toBeGreaterThanOrEqual(1);
      expect(op.execute([], {})).toEqual([]);
    });
  }
});
