import { describe, it, expect, vi } from 'vitest';
import { Tensor, SparseTensor } from '../src/ir/tensor.js';
import { unpackData, sparseToCoo } from '../src/sparse.js';
import { AbsOp, AddOp, ReluOp } from '../src/ops/index.ts';
import {
  SafeTensors,
  SafetensorsDuplicateKeyError,
  swapEndianness,
} from '../src/parser/safetensors.ts';
import * as safetensorsModule from '../src/parser/safetensors.ts';

// We mock the whole module to control getEndianness
vi.mock('../src/parser/safetensors.ts', async (importOriginal) => {
  const actual = (await importOriginal()) as any;
  return {
    ...actual,
    getEndianness: vi.fn().mockReturnValue('LE'), // Default to LE
  };
});

describe('Extra Coverage Gaps', () => {
  describe('sparse.ts fallback coverage', () => {
    it('unpackData fallback for non-specialized Uint8Array', () => {
      const data = new Uint8Array([10, 20]);
      const t = new Tensor('t', [2], 'int8', true, false, data);
      expect(unpackData(t)).toEqual([10, 20]);
    });

    it('unpackData fallback for non-iterable data', () => {
      const t = new Tensor('t', [1], 'float32', true, false, { notIterable: true } as any);
      expect(unpackData(t)).toEqual([]);
    });

    it('sparseToCoo fallback for unknown format', () => {
      const st = new SparseTensor('st', [1], 'COO');
      (st as any).format = 'UNKNOWN';
      expect(sparseToCoo(st)).toBe(st);
    });
  });

  describe('ops/index.ts non-Float32Array coverage', () => {
    it('AbsOp with Uint8Array data', () => {
      const op = new AbsOp();
      const data = new Float32Array([-1, 2, -3]);
      const input = new Tensor('x', [3], 'float32', true, false, new Uint8Array(data.buffer));
      const results = op.execute([input], {});
      expect(unpackData(results[0]!)).toEqual([1, 2, 3]);
    });

    it('AddOp with Uint8Array data', () => {
      const op = new AddOp();
      const dataA = new Float32Array([1, 2]);
      const dataB = new Float32Array([3, 4]);
      const inputA = new Tensor('a', [2], 'float32', true, false, new Uint8Array(dataA.buffer));
      const inputB = new Tensor('b', [2], 'float32', true, false, new Uint8Array(dataB.buffer));
      const results = op.execute([inputA, inputB], {});
      expect(unpackData(results[0]!)).toEqual([4, 6]);
    });

    it('ReluOp with Uint8Array data', () => {
      const op = new ReluOp();
      const data = new Float32Array([-1, 0, 1]);
      const input = new Tensor('x', [3], 'float32', true, false, new Uint8Array(data.buffer));
      const results = op.execute([input], {});
      expect(unpackData(results[0]!)).toEqual([0, 0, 1]);
    });
  });

  describe('safetensors.ts coverage', () => {
    it('should throw SafetensorsDuplicateKeyError when name is "toString"', () => {
      const header = {
        toString: { dtype: 'F32', shape: [1], data_offsets: [0, 4] },
      };
      const headerStr = JSON.stringify(header);
      const headerBytes = new TextEncoder().encode(headerStr);
      const buffer = new ArrayBuffer(8 + headerBytes.byteLength + 4);
      const view = new DataView(buffer);
      view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
      new Uint8Array(buffer, 8).set(headerBytes);

      expect(() => new SafeTensors(buffer)).toThrow(SafetensorsDuplicateKeyError);
    });

    it('should cover swapEndianness directly', () => {
      const data = new Uint8Array([0x01, 0x02, 0x03, 0x04]);
      swapEndianness(data.buffer, 0, 4, 2);
      expect(new Uint8Array(data.buffer)).toEqual(new Uint8Array([0x02, 0x01, 0x04, 0x03]));
    });

    it('should trigger swapEndianness in getTypedArray for BE', () => {
      // Control getEndianness via mocked function
      vi.mocked(safetensorsModule.getEndianness).mockReturnValue('BE');

      const header = {
        t: { dtype: 'I16', shape: [2], data_offsets: [0, 4] },
      };
      const headerStr = JSON.stringify(header);
      const headerBytes = new TextEncoder().encode(headerStr);
      const buffer = new ArrayBuffer(8 + headerBytes.byteLength + 8); // more space
      const view = new DataView(buffer);
      view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
      new Uint8Array(buffer, 8).set(headerBytes);

      // I16 data. Since system is LE (realistically),
      // SafeTensors sees system is BE (mocked) and file is LE, so it swaps.
      // If we put [0x02, 0x01], and it swaps, it becomes [0x01, 0x02].
      // [0x01, 0x02] as LE I16 is 0x0201 = 513.
      new Uint8Array(buffer, 8 + headerBytes.byteLength).set([0x02, 0x01, 0x04, 0x03]);

      // Pass 'BE' override to trigger the swap path regardless of host endianness
      const st = new SafeTensors(buffer, 'BE');
      const arr = st.getTypedArray('t') as Int16Array;

      // Verification:
      // Input bytes: [0x02, 0x01]
      // Swap (BE mode): [0x01, 0x02]
      // LE I16 view: 0x0201 = 513.
      expect(arr[0]).toBe(513);

      vi.mocked(safetensorsModule.getEndianness).mockReturnValue('LE');
    });
  });
});
