import { describe, expect, it } from 'vitest';
import { TemperatureLogitProcessor, TopKLogitProcessor } from '../src/genai/logit_processors.js';
import { Tensor } from '../src/ir/tensor.js';

describe('logit processors', () => {
  it('should process temperature', () => {
    const p = new TemperatureLogitProcessor(0.5);
    const logits = new Tensor('logits', [2], 'float32', false, false, new Float32Array([1.0, 2.0]));
    const res = p.process([], logits);
    expect((res.data as Float32Array)[0]).toBe(2.0);
  });

  it('should process topk', () => {
    const p = new TopKLogitProcessor(1);
    const logits = new Tensor(
      'logits',
      [3],
      'float32',
      false,
      false,
      new Float32Array([1.0, 5.0, 2.0]),
    );
    const res = p.process([], logits);
    expect((res.data as Float32Array)[0]).toBe(-Infinity);
    expect((res.data as Float32Array)[1]).toBe(5.0);
  });
});
