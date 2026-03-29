import { describe, it, expect, vi } from 'vitest';
import { DiffusionPipeline } from '../src/pipeline';
import {
  DDIMScheduler,
  DDPMScheduler,
  EulerDiscreteScheduler,
  PNDMScheduler,
  LMSDiscreteScheduler,
  DPMSolverMultistepScheduler,
  DPMSolverSinglestepScheduler,
  KDPM2DiscreteScheduler,
  KDPM2AncestralDiscreteScheduler,
  HeunDiscreteScheduler,
  UniPCMultistepScheduler,
  EulerAncestralDiscreteScheduler,
} from '../src/schedulers';
import { PyTorchPCG, rand, randn } from '../src/utils';

// Mock fetch
global.fetch = vi.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ _class_name: 'StableDiffusionPipeline' }),
  }),
) as any;

describe('DiffusionPipeline Phase 1', () => {
  it('fromPretrained', async () => {
    const pipeline = await DiffusionPipeline.fromPretrained(
      'hf-internal-testing/tiny-stable-diffusion-torch',
    );
    expect(pipeline).toBeDefined();
    expect(pipeline.modelIndex._class_name).toBe('StableDiffusionPipeline');
  });

  it('async execution and memory API', async () => {
    const pipeline = await DiffusionPipeline.fromPretrained(
      'hf-internal-testing/tiny-stable-diffusion-torch',
    );
    pipeline.freeMemory();

    let calledSteps = 0;
    const callback = (step: number, timestep: number, latents: number[]) => {
      calledSteps++;
    };

    const latents = await pipeline.call('dummy', 2, undefined, callback);
    expect(calledSteps).toBe(2);
    expect(latents.length).toBe(1 * 4 * 64 * 64);
  });

  it('pipeline abort', async () => {
    const pipeline = await DiffusionPipeline.fromPretrained(
      'hf-internal-testing/tiny-stable-diffusion-torch',
    );
    const controller = new AbortController();
    const signal = controller.signal;
    controller.abort();

    await expect(pipeline.call('dummy', 2, undefined, undefined, signal)).rejects.toThrow(
      'Pipeline aborted.',
    );
  });
});

describe('Schedulers Phase 2', () => {
  it('all schedulers can step', () => {
    const schedulers = [
      new DDIMScheduler(),
      new DDPMScheduler(),
      new EulerDiscreteScheduler(),
      new PNDMScheduler(),
      new LMSDiscreteScheduler(),
      new DPMSolverMultistepScheduler(),
      new DPMSolverSinglestepScheduler(),
      new KDPM2DiscreteScheduler(),
      new KDPM2AncestralDiscreteScheduler(),
      new HeunDiscreteScheduler(),
      new UniPCMultistepScheduler(),
      new EulerAncestralDiscreteScheduler(),
    ];
    const gen = new PyTorchPCG(42);
    const sample = [0.1, -0.2, 0.5];
    const modelOut = [-0.1, 0.4, 0.0];

    for (const sch of schedulers) {
      sch.setTimesteps(10);
      expect(sch.timesteps.length).toBe(10);
      const prev = sch.step(modelOut, sch.timesteps[0], sample, gen);
      expect(prev.length).toBe(3);
    }
  });
});

describe('Utils', () => {
  it('prng is deterministic', () => {
    const gen1 = new PyTorchPCG(123);
    const gen2 = new PyTorchPCG(123);

    const r1 = rand([10], gen1);
    const r2 = rand([10], gen2);
    expect(r1).toEqual(r2);

    const r3 = randn([2, 5], gen1);
    const r4 = randn([2, 5], gen2);
    expect(r3).toEqual(r4);
  });
});
