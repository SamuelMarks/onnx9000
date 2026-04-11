import { test, expect, vi } from 'vitest';
import {
  toEmscriptenType,
  DType,
  validateOnnxShapesAndDtypes,
} from '../src/parser/safetensors.validator';
import { Graph } from '../src/ir/graph';
import { Tensor } from '../src/ir/tensor';
import { saveSafetensors } from '../src/parser/safetensors';

test('safetensors.validator toEmscriptenType coverage', () => {
  expect(toEmscriptenType(DType.FLOAT)).toBe('Float32Array');
  expect(toEmscriptenType(DType.DOUBLE)).toBe('Float64Array');
  expect(toEmscriptenType(DType.INT8)).toBe('Int8Array');
  expect(toEmscriptenType(DType.INT16)).toBe('Int16Array');
  expect(toEmscriptenType(DType.INT32)).toBe('Int32Array');
  expect(toEmscriptenType(DType.INT64)).toBe('BigInt64Array');
  expect(toEmscriptenType(DType.UINT8)).toBe('Uint8Array');
  expect(toEmscriptenType(DType.UINT16)).toBe('Uint16Array');
  expect(toEmscriptenType(DType.UINT32)).toBe('Uint32Array');
  expect(toEmscriptenType(DType.UINT64)).toBe('BigUint64Array');
  expect(toEmscriptenType(DType.BOOL)).toBe('Uint8Array');
  expect(toEmscriptenType(DType.FLOAT16)).toBe('Uint16Array');
  expect(toEmscriptenType(DType.BFLOAT16)).toBe('Uint16Array');
  expect(toEmscriptenType(999)).toBe('Unknown');
});

test('safetensors.validator validateOnnxShapesAndDtypes warnings', () => {
  const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

  const stBytes = saveSafetensors({
    t1: { data: new Float32Array([1, 2]), dtype: 'F32', shape: [2] },
    t2: { data: new Float32Array([1, 2]), dtype: 'F32', shape: [2] },
    t3: { data: new Float32Array([1, 2]), dtype: 'F32', shape: [2] },
  });

  const g = new Graph();
  // Shape mismatch
  const t1 = new Tensor('t1', 'float32', [3]);
  g.tensors['t1'] = t1;

  // DType mismatch
  const t2 = new Tensor('t2', 'int32', [2]);
  g.tensors['t2'] = t2;

  // Unknown type
  const t3 = new Tensor('t3', 'unknown_type', [2]);
  g.tensors['t3'] = t3;

  validateOnnxShapesAndDtypes(stBytes.buffer, g);

  expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Shape mismatch for t1'));
  expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('DType mismatch for t2'));
  expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('DType mismatch for t3'));

  warnSpy.mockRestore();
});
