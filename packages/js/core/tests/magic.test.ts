import { describe, it, expect } from 'vitest';
import { detectFormat } from '../src/parser/magic.js';

describe('detectFormat', () => {
  it('should return unknown for small files', async () => {
    const blob = new Blob([new Uint8Array([1, 2])]);
    expect(await detectFormat(blob)).toBe('unknown');
  });

  it('should detect tflite', async () => {
    const data = new Uint8Array([0, 0, 0, 0, 0x54, 0x46, 0x4c, 0x33]);
    const blob = new Blob([data]);
    expect(await detectFormat(blob)).toBe('tflite');
  });

  it('should detect h5', async () => {
    const data = new Uint8Array([0x89, 0x48, 0x44, 0x46, 0, 0, 0, 0]);
    const blob = new Blob([data]);
    expect(await detectFormat(blob)).toBe('h5');
  });

  it('should detect zip / pt from bytes and name', async () => {
    const data = new Uint8Array([0x50, 0x4b, 0x03, 0x04, 0, 0, 0, 0]);
    const file = new File([data], 'model.pt');
    expect(await detectFormat(file)).toBe('pt');

    // Name doesn't end in pt, but has zip signature
    const file2 = new File([data], 'model.unknown');
    expect(await detectFormat(file2)).toBe('unknown');

    // Blob has no name property
    const blob = new Blob([data]);
    expect(await detectFormat(blob)).toBe('unknown');
  });

  it('should detect safetensors from name', async () => {
    const data = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0]);
    const file = new File([data], 'model.safetensors');
    expect(await detectFormat(file)).toBe('safetensors');
  });

  it('should detect onnx from name', async () => {
    const data = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0]);
    const file = new File([data], 'model.onnx');
    expect(await detectFormat(file)).toBe('onnx');
  });

  it('should fallback to unknown', async () => {
    const data = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0]);
    const file = new File([data], 'model.bin');
    expect(await detectFormat(file)).toBe('unknown');
  });
});
