import {
  TypicalLogitProcessor,
  RepetitionPenaltyLogitProcessor,
  TemperatureLogitProcessor,
  LogitProcessorList,
} from '../../src/genai/logit_processors';
import { Tensor } from '../../src/ir/tensor';
import { test, expect } from 'vitest';
import { describe, expect, it } from 'vitest';
import { Tensor } from '../../src/index.js';
import {
  TemperatureLogitProcessor,
  TopKLogitProcessor,
  RepetitionPenaltyLogitProcessor,
  MinPLogitProcessor,
  PresencePenaltyLogitProcessor,
  FrequencyPenaltyLogitProcessor,
  LogitProcessorList,
  ForcedBOSLogitProcessor,
  ForcedEOSLogitProcessor,
  LogitBiasProcessor,
  NoRepeatNGramLogitProcessor,
  NoBadWordsLogitProcessor,
  AllowedWordsLogitProcessor,
} from '../../src/genai/logit_processors.js';
import { TopPLogitProcessor } from '../../src/genai/top_p.js';

function createLogits(vals: number[]): Tensor {
  const data = new Float32Array(vals);
  return new Tensor('logits', [1, vals.length], 1, false, false, data);
}

describe('Logit Processors', () => {
  it('Temperature', () => {
    const proc = new TemperatureLogitProcessor(2.0);
    const logits = createLogits([2.0, 4.0]);
    const out = proc.process([], logits);
    const data = out.data as Float32Array;
    expect(data[0]).toBe(1.0);
    expect(data[1]).toBe(2.0);
  });

  it('TopK', () => {
    const proc = new TopKLogitProcessor(2);
    const logits = createLogits([1.0, 5.0, 3.0, 2.0]);
    const out = proc.process([], logits);
    const data = out.data as Float32Array;
    expect(data[0]).toBe(-Infinity);
    expect(data[1]).toBe(5.0);
    expect(data[2]).toBe(3.0);
    expect(data[3]).toBe(-Infinity);
  });

  it('Repetition Penalty', () => {
    const proc = new RepetitionPenaltyLogitProcessor(2.0);
    const logits = createLogits([1.0, -1.0, 3.0]);
    const out = proc.process([1], logits);
    const data = out.data as Float32Array;
    expect(data[0]).toBe(1.0);
    expect(data[1]).toBe(-2.0);
    expect(data[2]).toBe(3.0);
  });

  it('TopP', () => {
    const proc = new TopPLogitProcessor(0.9);
    const logits = createLogits([1.0, 9.0, 10.0]);
    const out = proc.process([], logits);
    const data = out.data as Float32Array;
    expect(data[0]).toBe(-Infinity);
    expect(data[1]).toBe(9.0);
    expect(data[2]).toBe(10.0);
  });

  it('LogitProcessorList', () => {
    const lst = new LogitProcessorList([
      new TemperatureLogitProcessor(2.0),
      new TopKLogitProcessor(1),
    ]);
    const logits = createLogits([2.0, 4.0]);
    const out = lst.process([], logits);
    const data = out.data as Float32Array;
    expect(data[0]).toBe(-Infinity);
    expect(data[1]).toBe(2.0);
  });

  it('MinP', () => {
    const proc = new MinPLogitProcessor(0.1);
    const logits = createLogits([1.0, 8.0, 10.0]);
    const out = proc.process([], logits);
    const data = out.data as Float32Array;
    expect(data[0]).toBe(-Infinity);
    expect(data[1]).toBe(8.0);
    expect(data[2]).toBe(10.0);
  });

  it('Presence Penalty', () => {
    const proc = new PresencePenaltyLogitProcessor(1.5);
    const logits = createLogits([2.0, 4.0, 6.0]);
    const out = proc.process([1, 1], logits);
    const data = out.data as Float32Array;
    expect(data[0]).toBe(2.0);
    expect(data[1]).toBe(2.5); // 4.0 - 1.5
    expect(data[2]).toBe(6.0);
  });

  it('Frequency Penalty', () => {
    const proc = new FrequencyPenaltyLogitProcessor(1.5);
    const logits = createLogits([2.0, 4.0, 6.0]);
    const out = proc.process([1, 1], logits);
    const data = out.data as Float32Array;
    expect(data[0]).toBe(2.0);
    expect(data[1]).toBe(1.0); // 4.0 - 1.5 * 2
    expect(data[2]).toBe(6.0);
  });

  it('ForcedBOS', () => {
    const proc = new ForcedBOSLogitProcessor(1);
    const logits = createLogits([2.0, 4.0, 6.0]);
    const out = proc.process([], logits);
    const data = out.data as Float32Array;
    expect(data[0]).toBe(-Infinity);
    expect(data[1]).toBe(4.0);
    expect(data[2]).toBe(-Infinity);
  });

  it('NoRepeatNGram', () => {
    const proc = new NoRepeatNGramLogitProcessor(2);
    const logits = createLogits([2.0, 4.0, 6.0]);
    const out = proc.process([0, 1, 0], logits);
    const data = out.data as Float32Array;
    expect(data[0]).toBe(2.0);
    expect(data[1]).toBe(-Infinity);
    expect(data[2]).toBe(6.0);
  });

  it('NoBadWords', () => {
    const proc = new NoBadWordsLogitProcessor([[1, 2]]);
    const logits = createLogits([2.0, 4.0, 6.0]);
    const out = proc.process([1], logits);
    const data = out.data as Float32Array;
    expect(data[0]).toBe(2.0);
    expect(data[1]).toBe(4.0);
    expect(data[2]).toBe(-Infinity);
  });

  it('AllowedWords', () => {
    const proc = new AllowedWordsLogitProcessor([0, 2]);
    const logits = createLogits([2.0, 4.0, 6.0]);
    const out = proc.process([], logits);
    const data = out.data as Float32Array;
    expect(data[0]).toBe(2.0);
    expect(data[1]).toBe(-Infinity);
    expect(data[2]).toBe(6.0);
  });
});

test('TypicalLogitProcessor coverage', async () => {
  const p = new TypicalLogitProcessor();
  const t = new Tensor('t', 'float32', [1]);
  expect(p.process([], t)).toBe(t);
});

test('RepetitionPenaltyLogitProcessor val > 0', async () => {
  const p = new RepetitionPenaltyLogitProcessor(1.2);

  const t = new Tensor('t', 'float32', [2]);
  t.data = new Float32Array([2.0, 2.0]);

  const out = p.process([0], t);
  expect(out.data[0]).toBeLessThan(2.0);

  // Also test negative
  t.data[0] = -2.0;
  const out2 = p.process([0], t);
  expect(out2.data[0]).toBeLessThan(-2.0);
});

test('Processor non-float32 return early', async () => {
  const p = new RepetitionPenaltyLogitProcessor(1.2);
  const l = new LogitProcessorList([p]);
  const t = new Tensor('t', 'int32', [1]);
  expect(l.process([], t)).toBe(t);
});

test('TemperatureLogitProcessor non-float32', async () => {
  const p = new TemperatureLogitProcessor(1.2);
  const t = new Tensor('t', 'int32', [1]);
  expect(p.process([], t)).toBe(t);
});
