import { PyTorchPCG, parseModelIndex } from './utils';
import { UNet2DConditionModel, AutoencoderKL } from './models';
import { Scheduler, DDPMScheduler } from './schedulers';

export class DiffusionPipeline {
  /** Configuration object for the pipeline. */
  config: Record<string, any>;
  /** Device to run the models on. */
  device: string;
  /** Index data of the loaded models. */
  modelIndex: any;
  /** UNet model. */
  unet: UNet2DConditionModel;
  /** VAE autoencoder. */
  vae: AutoencoderKL;
  /** Scheduler. */
  scheduler: Scheduler;

  private _isAborted: boolean = false;

  /**
   * Initialize a DiffusionPipeline.
   * @param config - Configuration options.
   */
  constructor(config: Record<string, any> = {}) {
    this.config = config;
    this.device = 'cpu';
    this.unet = new UNet2DConditionModel();
    this.vae = new AutoencoderKL();
    this.scheduler = new DDPMScheduler();
  }

  /**
   * Load pipeline from Hugging Face model id.
   * @param modelId - The Hugging Face repo ID.
   * @returns A new DiffusionPipeline instance.
   */
  static async fromPretrained(modelId: string): Promise<DiffusionPipeline> {
    const pipeline = new DiffusionPipeline({ model_path: modelId });
    try {
      pipeline.modelIndex = await parseModelIndex(modelId);
    } catch (e) {
      pipeline.modelIndex = {};
    }
    return pipeline;
  }

  /**
   * Run the diffusion pipeline.
   * @param prompt - Text prompt (currently ignored).
   * @param numInferenceSteps - Number of denoising steps.
   * @param generator - Optional random number generator.
   * @param callback - Optional callback function to track progress.
   * @param signal - Optional AbortSignal.
   * @returns Denoised latent vector (or image).
   */
  async call(
    prompt: string,
    numInferenceSteps: number = 50,
    generator?: PyTorchPCG,
    callback?: (step: number, timestep: number, latents: Float32Array) => void,
    signal?: AbortSignal,
  ): Promise<Float32Array> {
    this._isAborted = false;
    if (signal) {
      signal.addEventListener('abort', () => {
        this._isAborted = true;
      });
      if (signal.aborted) {
        this._isAborted = true;
      }
    }

    const gen = generator || new PyTorchPCG(42);
    this.scheduler.setTimesteps(numInferenceSteps);

    // Initial random noise
    const latentSize = 64 * 64 * 4;
    let latents = new Float32Array(latentSize);
    for (let i = 0; i < latentSize; i++) {
      latents[i] = gen.nextFloat() * 2 - 1.0;
    }

    const encoder_hidden_states = new Float32Array(77 * 768).fill(0.1);

    for (let step = 0; step < numInferenceSteps; step++) {
      if (this._isAborted) {
        throw new Error('Pipeline aborted.');
      }

      const timestep = this.scheduler.timesteps[step] || 0;
      const noise_pred = this.unet.call(latents, timestep, encoder_hidden_states);
      latents = this.scheduler.step(noise_pred, timestep, latents, gen) as any;

      if (callback) {
        callback(step, timestep, latents);
      }
      await new Promise((r) => setTimeout(r, 0));
    }

    const decoded = this.vae.decode(latents);
    return decoded;
  }

  /**
   * Free memory associated with models.
   */
  freeMemory(): void {
    this._isAborted = true;
  }
}
