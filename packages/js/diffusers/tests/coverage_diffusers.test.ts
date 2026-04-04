import { describe, it, expect, vi } from 'vitest';
import * as idx from '../src/index';
import * as models from '../src/models';
import * as reg from '../src/registry';
import * as scheds from '../src/schedulers';
import * as utils from '../src/utils';

describe('Diffusers coverage', () => {
  it('index', () => {
    expect(idx).toBeDefined();
  });

  it('models', () => {
    const vae = new models.AutoencoderKL();
    const f32 = new Float32Array([1, 2]);
    const enc = vae.encode(f32);
    expect(enc[0]).toBeCloseTo(0.18215);
    const dec = vae.decode(f32);
    expect(dec[0]).toBeCloseTo(1 / 0.18215);

    const unet = new models.UNet2DConditionModel();
    const out = unet.call(f32, 10, new Float32Array([1]));
    expect(out[0]).toBeCloseTo(1 - 0.1);
  });

  it('registry', () => {
    expect(reg).toBeDefined();
  });

  it('schedulers empty subclasses', () => {
    expect(new scheds.PNDMScheduler()).toBeDefined();
    expect(new scheds.LMSDiscreteScheduler()).toBeDefined();
    expect(new scheds.DPMSolverMultistepScheduler()).toBeDefined();
    expect(new scheds.DPMSolverSinglestepScheduler()).toBeDefined();
    expect(new scheds.KDPM2DiscreteScheduler()).toBeDefined();
    expect(new scheds.KDPM2AncestralDiscreteScheduler()).toBeDefined();
    expect(new scheds.HeunDiscreteScheduler()).toBeDefined();
    expect(new scheds.UniPCMultistepScheduler()).toBeDefined();
    expect(new scheds.EulerAncestralDiscreteScheduler()).toBeDefined();
  });

  it('LCMScheduler', () => {
    const s = new scheds.LCMScheduler(10);
    const out = s.step([1], 5, [2]);
    expect(out[0]).toBe(1);

    const outF = s.step(new Float32Array([1]), 5, new Float32Array([2]));
    expect(outF[0]).toBe(1);
  });

  it('registry decorator', () => {
    @reg.register_op('domain', 'name')
    class TestClass {}
    expect((TestClass as any).domain).toBe('domain');
  });

  it('pipeline operations', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ a: 1 }),
    });
    vi.stubGlobal('fetch', mockFetch);

    const p = await idx.DiffusionPipeline.fromPretrained('repo');
    expect(p.modelIndex).toEqual({ a: 1 });

    // test callback and normal call
    const cb = vi.fn();
    const callP = await p.call('test', 2, undefined, cb);
    expect(cb).toHaveBeenCalledTimes(2);

    // test abort listener
    const ac = new AbortController();
    const callP2 = p.call('test', 10, undefined, undefined, ac.signal);
    ac.abort();
    await expect(callP2).rejects.toThrow('Pipeline aborted.');

    // test abort already signaled
    const ac2 = new AbortController();
    ac2.abort();
    await expect(p.call('test', 1, undefined, undefined, ac2.signal)).rejects.toThrow(
      'Pipeline aborted.',
    );

    // test free memory
    p.freeMemory();
    expect((p as any)._isAborted).toBe(true);

    vi.unstubAllGlobals();
  });

  it('utils - setProgressBarConfig', () => {
    utils.setProgressBarConfig(false);
    expect(utils.globalProgressBarConfig.enabled).toBe(false);
  });

  it('utils - fetchHubFile error', async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: false });
    vi.stubGlobal('fetch', mockFetch);
    await expect(utils.fetchHubFile('repo', 'file')).rejects.toThrow('Failed to fetch');
    vi.unstubAllGlobals();
  });
});
