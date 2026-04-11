import { describe, it, expect, vi } from 'vitest';
import {
  mapOnnxTypeToTflite,
  mapOnnxShapeToTflite,
  createShapeSignature,
} from '../src/compiler/mapping';
import { TensorType } from '../src/flatbuffer/schema';

describe('mapping', () => {
  it('mapOnnxTypeToTflite', () => {
    expect(mapOnnxTypeToTflite('float16')).toBe(TensorType.FLOAT16);
    expect(mapOnnxTypeToTflite('int32')).toBe(TensorType.INT32);
    expect(mapOnnxTypeToTflite('int8')).toBe(TensorType.INT8);
    expect(mapOnnxTypeToTflite('uint8')).toBe(TensorType.UINT8);
    expect(mapOnnxTypeToTflite('bool')).toBe(TensorType.BOOL);
    expect(mapOnnxTypeToTflite('string')).toBe(TensorType.STRING);
    expect(mapOnnxTypeToTflite('float64')).toBe(TensorType.FLOAT32);
    expect(mapOnnxTypeToTflite('unknown' as Object)).toBe(TensorType.FLOAT32);
  });

  it('mapOnnxShapeToTflite', () => {
    expect(mapOnnxShapeToTflite([])).toEqual([]);
    expect(mapOnnxShapeToTflite([1, -1, 3])).toEqual([1, -1, 3]);
    expect(mapOnnxShapeToTflite(['batch' as Object, 3])).toEqual([-1, 3]);
  });

  it('createShapeSignature', () => {
    expect(createShapeSignature([1, 2, 3])).toEqual([1, 2, 3]);
  });
});

describe('mapping - extra', () => {
  it('mapOnnxTypeToTflite int64 and float32', () => {
    const mockWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    expect(mapOnnxTypeToTflite('int64', 'test_tensor')).toBe(TensorType.INT32);
    expect(mockWarn).toHaveBeenCalled();
    mockWarn.mockRestore();

    expect(mapOnnxTypeToTflite('float32')).toBe(TensorType.FLOAT32);
  });
});
