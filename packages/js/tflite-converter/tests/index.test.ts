import { describe, expect, it } from 'vitest';
import { Tfliteconverter } from '../src/index.js';

describe('Tfliteconverter', () => {
  it('should emit and verify TFL3 FlatBuffer headers', () => {
    const conv = new Tfliteconverter();
    expect(conv.run()).toBe('[tflite-converter] processed');
    expect(conv.hasTfl3Header(conv.emitHeader())).toBe(true);
    expect(conv.hasTfl3Header(new Uint8Array(4))).toBe(false);
    expect(conv.hasTfl3Header(new Uint8Array(8))).toBe(false);
  });
});
