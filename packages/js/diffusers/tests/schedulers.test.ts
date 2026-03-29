import { describe, it, expect } from 'vitest';
import { DDIMScheduler, EulerDiscreteScheduler } from '../src/schedulers';

describe('Schedulers', () => {
  it('DDIM step', () => {
    const scheduler = new DDIMScheduler(1000);
    const sample = new Float32Array([1.0]);
    const modelOutput = new Float32Array([0.1]);
    const timestep = 999;

    const out = scheduler.step(modelOutput, timestep, sample);
    expect(out.length).toBe(1);
  });

  it('Euler step', () => {
    const scheduler = new EulerDiscreteScheduler(1000);
    const sample = new Float32Array([1.0]);
    const modelOutput = new Float32Array([0.1]);
    const timestep = 500;

    const out = scheduler.step(modelOutput, timestep, sample);
    expect(out.length).toBe(1);
    expect(out[0]).toBeCloseTo(1.05);
  });
});
