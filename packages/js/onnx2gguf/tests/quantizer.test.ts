import { expect, test } from 'vitest';
import { f32ToF16, quantizeQ4_0, quantizeQ4_1, quantizeQ8_0 } from '../src/quantizer';

test('quantizer', () => {
  const floats = new Float32Array(32);
  for (let i = 0; i < 32; i++) floats[i] = i;
  const data = new Uint8Array(floats.buffer);

  const f16 = f32ToF16(data);
  expect(f16.byteLength).toBe(64);

  const q4_0 = quantizeQ4_0(data);
  expect(q4_0.byteLength).toBe(18);

  const q4_1 = quantizeQ4_1(data);
  expect(q4_1.byteLength).toBe(20);

  const q8_0 = quantizeQ8_0(data);
  expect(q8_0.byteLength).toBe(34);

  const badData = new Uint8Array(new Float32Array(30).buffer);
  expect(() => quantizeQ4_0(badData)).toThrow('Q4_0 requires multiples of 32');
  expect(() => quantizeQ4_1(badData)).toThrow('Q4_1 requires multiples of 32');
  expect(() => quantizeQ8_0(badData)).toThrow('Q8_0 requires multiples of 32');
});
