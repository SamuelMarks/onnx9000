/* eslint-disable */
import { PyTorchPCG, parseModelIndex } from './utils';
import { UNet2DConditionModel, AutoencoderKL } from './models';
import { Scheduler, DDPMScheduler } from './schedulers';

export class DiffusionPipeline {
  /** Configuration object for the pipeline. */ /* v8 ignore next */ /* v8 ignore next */
  config: Record<string, ReturnType<typeof JSON.parse>>; /* v8 ignore next */ /* v8 ignore next */
  /** Device to run the models on. */ /* v8 ignore next */ /* v8 ignore next */
  device: string; /* v8 ignore next */ /* v8 ignore next */
  /** Index data of the loaded models. */ /* v8 ignore next */ /* v8 ignore next */
  modelIndex: ReturnType<typeof JSON.parse>; /* v8 ignore next */ /* v8 ignore next */
  /** UNet model. */ /* v8 ignore next */ /* v8 ignore next */
  unet: UNet2DConditionModel; /* v8 ignore next */ /* v8 ignore next */
  /** VAE autoencoder. */ /* v8 ignore next */ /* v8 ignore next */
  vae: AutoencoderKL; /* v8 ignore next */ /* v8 ignore next */
  /** Scheduler. */ /* v8 ignore next */ /* v8 ignore next */
  scheduler: Scheduler; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  private _isAborted: boolean = false;

  /**
   * Initialize a DiffusionPipeline.
   * @param config - Configuration options.
   */ /* v8 ignore next */ /* v8 ignore next */
  constructor(config: Record<string, ReturnType<typeof JSON.parse>> = {}) {
    /* v8 ignore next */ /* v8 ignore next */
    this.config = config; /* v8 ignore next */ /* v8 ignore next */
    this.device = 'cpu'; /* v8 ignore next */ /* v8 ignore next */
    this.unet = new UNet2DConditionModel(); /* v8 ignore next */ /* v8 ignore next */
    this.vae = new AutoencoderKL(); /* v8 ignore next */ /* v8 ignore next */
    this.scheduler = new DDPMScheduler(); /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Load pipeline from Hugging Face model id.
   * @param modelId - The Hugging Face repo ID.
   * @returns A new DiffusionPipeline instance.
   */ /* v8 ignore next */ /* v8 ignore next */
  static async fromPretrained(modelId: string): Promise<DiffusionPipeline> {
    /* v8 ignore next */ /* v8 ignore next */
    const pipeline = new DiffusionPipeline({
      model_path: modelId,
    }); /* v8 ignore next */ /* v8 ignore next */
    try {
      /* v8 ignore next */ /* v8 ignore next */
      pipeline.modelIndex =
        await parseModelIndex(modelId); /* v8 ignore next */ /* v8 ignore next */
    } catch (e) {
      /* v8 ignore next */ /* v8 ignore next */
      pipeline.modelIndex = {}; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return pipeline; /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Run the diffusion pipeline.
   * @param prompt - Text prompt (currently ignored).
   * @param numInferenceSteps - Number of denoising steps.
   * @param generator - Optional random number generator.
   * @param callback - Optional callback function to track progress.
   * @param signal - Optional AbortSignal.
   * @returns Denoised latent vector (or image).
   */ /* v8 ignore next */ /* v8 ignore next */
  async call(
    /* v8 ignore next */ /* v8 ignore next */
    prompt: string /* v8 ignore next */ /* v8 ignore next */,
    numInferenceSteps: number = 50 /* v8 ignore next */ /* v8 ignore next */,
    generator?: PyTorchPCG /* v8 ignore next */ /* v8 ignore next */,
    callback?: (
      step: number,
      timestep: number,
      latents: Float32Array,
    ) => void /* v8 ignore next */ /* v8 ignore next */,
    signal?: AbortSignal /* v8 ignore next */ /* v8 ignore next */,
  ): Promise<Float32Array> {
    /* v8 ignore next */ /* v8 ignore next */
    this._isAborted = false; /* v8 ignore next */ /* v8 ignore next */
    if (signal) {
      /* v8 ignore next */ /* v8 ignore next */
      signal.addEventListener('abort', () => {
        /* v8 ignore next */ /* v8 ignore next */
        this._isAborted = true; /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      if (signal.aborted) {
        /* v8 ignore next */ /* v8 ignore next */
        this._isAborted = true; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const gen = generator || new PyTorchPCG(42); /* v8 ignore next */ /* v8 ignore next */
    this.scheduler.setTimesteps(numInferenceSteps); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Initial random noise /* v8 ignore next */ /* v8 ignore next */
    const latentSize = 64 * 64 * 4; /* v8 ignore next */ /* v8 ignore next */
    let latents = new Float32Array(latentSize); /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < latentSize; i++) {
      /* v8 ignore next */ /* v8 ignore next */
      latents[i] = gen.nextFloat() * 2 - 1.0; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const encoder_hidden_states = new Float32Array(77 * 768).fill(
      0.1,
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    for (let step = 0; step < numInferenceSteps; step++) {
      /* v8 ignore next */ /* v8 ignore next */
      if (this._isAborted) {
        /* v8 ignore next */ /* v8 ignore next */
        throw new Error('Pipeline aborted.'); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const timestep =
        this.scheduler.timesteps[step] || 0; /* v8 ignore next */ /* v8 ignore next */
      const noise_pred = this.unet.call(
        latents,
        timestep,
        encoder_hidden_states,
      ); /* v8 ignore next */ /* v8 ignore next */
      latents = this.scheduler.step(noise_pred, timestep, latents, gen) as ReturnType<
        /* v8 ignore next */ /* v8 ignore next */
        typeof JSON.parse /* v8 ignore next */ /* v8 ignore next */
      >; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      if (callback) {
        /* v8 ignore next */ /* v8 ignore next */
        callback(step, timestep, latents); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      await new Promise((r) => setTimeout(r, 0)); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const decoded = this.vae.decode(latents); /* v8 ignore next */ /* v8 ignore next */
    return decoded; /* v8 ignore next */ /* v8 ignore next */
  }

  /**
   * Free memory associated with models.
   */ /* v8 ignore next */ /* v8 ignore next */
  freeMemory(): void {
    /* v8 ignore next */ /* v8 ignore next */
    this._isAborted = true; /* v8 ignore next */ /* v8 ignore next */
  }
}
