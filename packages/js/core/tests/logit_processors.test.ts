import { describe, expect, it } from 'vitest';
import { TemperatureLogitProcessor, TopKLogitProcessor } from '../src/genai/logit_processors.js';

describe('logit processors', () => {
  it('should process temperature', () => {
    const p = new TemperatureLogitProcessor(0.5);
    const logits: any = { data: new Float32Array([1.0, 2.0]), shape: [2] };
    const res = p.process([], logits);
    expect(res.data[0]).toBe(2.0);
  });

  it('should process topk', () => {
    const p = new TopKLogitProcessor(1);
    const logits: any = { data: new Float32Array([1.0, 5.0, 2.0]), shape: [3] };
    const res = p.process([], logits);
    expect(res.data[0]).toBe(-Infinity);
    expect(res.data[1]).toBe(5.0);
  });
});
