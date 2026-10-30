import { describe, expect, it } from 'vitest';
import { ORTTraining } from '../src/index.js';

describe('ORTTraining', () => {
  it('should execute SGD parameter updates and validate lengths', () => {
    const trainer = new ORTTraining();
    expect(trainer.process('epoch_1')).toBe('ORT Training processed epoch_1');

    const updated = trainer.stepSgd(new Float32Array([2.0, 4.0]), new Float32Array([0.5, 1.0]), {
      learningRate: 0.1,
      weightDecay: 0.1,
    });
    expect(updated[0]).toBeCloseTo(2.0 - 0.1 * (0.5 + 0.2));
    expect(updated[1]).toBeCloseTo(4.0 - 0.1 * (1.0 + 0.4));

    expect(() =>
      trainer.stepSgd(new Float32Array([1.0]), new Float32Array([1.0, 2.0]), {
        learningRate: 0.1,
      }),
    ).toThrow('Weights and gradients must have matching lengths');
  });
});
