/**
 * @fileoverview schedulers.ts
 * Provides schedulers functionality for the diffusers package.
 */
import type { PyTorchPCG } from './utils';

/** Represents the `Scheduler` class. */
export class Scheduler {
  /** Property `numTrainTimesteps` of `Scheduler`. */
  numTrainTimesteps: number;
  /** Property `timesteps` of `Scheduler`. */
  timesteps: number[];

  /**
   * Initializes a new `Scheduler` instance.
   * @param numTrainTimesteps Parameter `numTrainTimesteps`.
   */
  constructor(numTrainTimesteps: number = 1000) {
    this.numTrainTimesteps = numTrainTimesteps;
    this.timesteps = Array.from({ length: numTrainTimesteps }, (_, i) => numTrainTimesteps - 1 - i);
  }

  /**
   * Executes `setTimesteps` on `Scheduler`.
   * @param numInferenceSteps Parameter `numInferenceSteps`.
   */
  setTimesteps(numInferenceSteps: number): void {
    const step = Math.floor(this.numTrainTimesteps / numInferenceSteps);
    this.timesteps = Array.from(
      { length: numInferenceSteps },
      (_, i) => this.numTrainTimesteps - 1 - i * step,
    );
  }

  /**
   * Executes `step` on `Scheduler`.
   * @param _modelOutput Parameter `_modelOutput`.
   * @param _timestep Parameter `_timestep`.
   * @param sample Parameter `sample`.
   * @param _gen Parameter `_gen`.
   * @returns Result of `step`.
   */
  step(
    _modelOutput: Float32Array | number[],
    _timestep: number,
    sample: Float32Array | number[],
    _gen?: PyTorchPCG,
  ): Float32Array | number[] {
    return sample;
  }
}

/** Represents the `DDIMScheduler` class. */
export class DDIMScheduler extends Scheduler {
  /**
   * Executes `step` on `DDIMScheduler`.
   * @param modelOutput Parameter `modelOutput`.
   * @param timestep Parameter `timestep`.
   * @param sample Parameter `sample`.
   * @param _gen Parameter `_gen`.
   * @returns Result of `step`.
   */
  override step(
    modelOutput: Float32Array | number[],
    timestep: number,
    sample: Float32Array | number[],
    _gen?: PyTorchPCG,
  ): Float32Array | number[] {
    const isArray = Array.isArray(sample);
    const out = isArray ? new Array(sample.length) : new Float32Array(sample.length);
    const alphaProdT = 1.0 - timestep / this.numTrainTimesteps;
    const betaProdT = 1 - alphaProdT;
    for (let i = 0; i < sample.length; i++) {
      out[i] =
        ((sample[i] as number) - Math.sqrt(betaProdT) * (modelOutput[i] as number)) /
        Math.sqrt(alphaProdT);
    }
    return out;
  }
}

/** Represents the `DDPMScheduler` class. */
export class DDPMScheduler extends Scheduler {
  /**
   * Executes `step` on `DDPMScheduler`.
   * @param modelOutput Parameter `modelOutput`.
   * @param timestep Parameter `timestep`.
   * @param sample Parameter `sample`.
   * @param _gen Parameter `_gen`.
   * @returns Result of `step`.
   */
  override step(
    modelOutput: Float32Array | number[],
    timestep: number,
    sample: Float32Array | number[],
    _gen?: PyTorchPCG,
  ): Float32Array | number[] {
    const isArray = Array.isArray(sample);
    const out = isArray ? new Array(sample.length) : new Float32Array(sample.length);
    const alphaT = 1.0 - timestep / this.numTrainTimesteps;
    for (let i = 0; i < sample.length; i++) {
      out[i] =
        ((sample[i] as number) - (1 - alphaT) * (modelOutput[i] as number)) / Math.sqrt(alphaT);
    }
    return out;
  }
}

/** Represents the `EulerDiscreteScheduler` class. */
export class EulerDiscreteScheduler extends Scheduler {
  /**
   * Executes `step` on `EulerDiscreteScheduler`.
   * @param modelOutput Parameter `modelOutput`.
   * @param timestep Parameter `timestep`.
   * @param sample Parameter `sample`.
   * @param _gen Parameter `_gen`.
   * @returns Result of `step`.
   */
  override step(
    modelOutput: Float32Array | number[],
    timestep: number,
    sample: Float32Array | number[],
    _gen?: PyTorchPCG,
  ): Float32Array | number[] {
    const isArray = Array.isArray(sample);
    const out = isArray ? new Array(sample.length) : new Float32Array(sample.length);
    const sigma = timestep / this.numTrainTimesteps;
    for (let i = 0; i < sample.length; i++) {
      out[i] = (sample[i] as number) + (modelOutput[i] as number) * sigma;
    }
    return out;
  }
}

/** Represents the `LCMScheduler` class. */
export class LCMScheduler extends Scheduler {
  /**
   * Executes `step` on `LCMScheduler`.
   * @param modelOutput Parameter `modelOutput`.
   * @param _timestep Parameter `_timestep`.
   * @param sample Parameter `sample`.
   * @param _gen Parameter `_gen`.
   * @returns Result of `step`.
   */
  override step(
    modelOutput: Float32Array | number[],
    _timestep: number,
    sample: Float32Array | number[],
    _gen?: PyTorchPCG,
  ): Float32Array | number[] {
    const isArray = Array.isArray(sample);
    const out = isArray ? new Array(sample.length) : new Float32Array(sample.length);
    for (let i = 0; i < sample.length; i++) {
      out[i] = (sample[i] as number) - (modelOutput[i] as number);
    }
    return out;
  }
}

/** Represents the `PNDMScheduler` class. */
export class PNDMScheduler extends Scheduler {}
/** Represents the `LMSDiscreteScheduler` class. */
export class LMSDiscreteScheduler extends Scheduler {}
/** Represents the `DPMSolverMultistepScheduler` class. */
export class DPMSolverMultistepScheduler extends Scheduler {}
/** Represents the `DPMSolverSinglestepScheduler` class. */
export class DPMSolverSinglestepScheduler extends Scheduler {}
/** Represents the `KDPM2DiscreteScheduler` class. */
export class KDPM2DiscreteScheduler extends Scheduler {}
/** Represents the `KDPM2AncestralDiscreteScheduler` class. */
export class KDPM2AncestralDiscreteScheduler extends Scheduler {}
/** Represents the `HeunDiscreteScheduler` class. */
export class HeunDiscreteScheduler extends Scheduler {}
/** Represents the `UniPCMultistepScheduler` class. */
export class UniPCMultistepScheduler extends Scheduler {}
/** Represents the `EulerAncestralDiscreteScheduler` class. */
export class EulerAncestralDiscreteScheduler extends Scheduler {}
