import { PyTorchPCG, parseModelIndex } from './utils';

export class DiffusionPipeline {
  config: Record<string, any>;
  device: string;
  modelIndex: any;
  private _isAborted: boolean = false;

  constructor(config: Record<string, any> = {}) {
    this.config = config;
    this.device = 'cpu';
  }

  static async fromPretrained(modelId: string): Promise<DiffusionPipeline> {
    const pipeline = new DiffusionPipeline({ model_path: modelId });
    pipeline.modelIndex = await parseModelIndex(modelId);
    return pipeline;
  }

  async call(
    prompt: string,
    numInferenceSteps: number = 50,
    generator?: PyTorchPCG,
    callback?: (step: number, timestep: number, latents: number[]) => void,
    signal?: AbortSignal,
  ): Promise<number[]> {
    this._isAborted = false;
    if (signal) {
      signal.addEventListener('abort', () => {
        this._isAborted = true;
      });
      if (signal.aborted) {
        this._isAborted = true;
      }
    }

    const latents = new Array(64 * 64 * 4).fill(0.0);
    for (let step = 0; step < numInferenceSteps; step++) {
      if (this._isAborted) {
        throw new Error('Pipeline aborted.');
      }
      for (let i = 0; i < latents.length; i++) {
        latents[i] *= 0.9;
      }
      if (callback) {
        callback(step, step, latents);
      }
      await new Promise((r) => setTimeout(r, 0));
    }
    return latents;
  }

  freeMemory(): void {
    this._isAborted = true;
  }
}
