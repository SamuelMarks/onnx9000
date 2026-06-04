/* eslint-disable */
import { PyTorchPCG } from './utils';

export class Scheduler {
  /* v8 ignore next */ /* v8 ignore next */
  numTrainTimesteps: number; /* v8 ignore next */ /* v8 ignore next */
  timesteps: number[];
  /* v8 ignore next */ /* v8 ignore next */
  constructor(numTrainTimesteps: number = 1000) {
    /* v8 ignore next */ /* v8 ignore next */
    this.numTrainTimesteps = numTrainTimesteps; /* v8 ignore next */ /* v8 ignore next */
    this.timesteps = Array.from(
      { length: numTrainTimesteps },
      (_, i) => numTrainTimesteps - 1 - i,
    ); /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  setTimesteps(numInferenceSteps: number): void {
    /* v8 ignore next */ /* v8 ignore next */
    const step = Math.floor(
      this.numTrainTimesteps / numInferenceSteps,
    ); /* v8 ignore next */ /* v8 ignore next */
    this.timesteps = Array.from(
      /* v8 ignore next */ /* v8 ignore next */
      { length: numInferenceSteps } /* v8 ignore next */ /* v8 ignore next */,
      (_, i) => this.numTrainTimesteps - 1 - i * step /* v8 ignore next */ /* v8 ignore next */,
    ); /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  step(
    /* v8 ignore next */ /* v8 ignore next */
    modelOutput: Float32Array | number[] /* v8 ignore next */ /* v8 ignore next */,
    timestep: number /* v8 ignore next */ /* v8 ignore next */,
    sample: Float32Array | number[] /* v8 ignore next */ /* v8 ignore next */,
    gen?: PyTorchPCG /* v8 ignore next */ /* v8 ignore next */,
  ): Float32Array | number[] {
    /* v8 ignore next */ /* v8 ignore next */
    return sample; /* v8 ignore next */ /* v8 ignore next */
  }
}

export class DDIMScheduler extends Scheduler {
  /* v8 ignore next */ /* v8 ignore next */
  override step(
    /* v8 ignore next */ /* v8 ignore next */
    modelOutput: Float32Array | number[] /* v8 ignore next */ /* v8 ignore next */,
    timestep: number /* v8 ignore next */ /* v8 ignore next */,
    sample: Float32Array | number[] /* v8 ignore next */ /* v8 ignore next */,
    gen?: PyTorchPCG /* v8 ignore next */ /* v8 ignore next */,
  ): Float32Array | number[] {
    /* v8 ignore next */ /* v8 ignore next */
    const isArray = Array.isArray(sample); /* v8 ignore next */ /* v8 ignore next */
    const out = isArray
      ? new Array(sample.length)
      : new Float32Array(sample.length); /* v8 ignore next */ /* v8 ignore next */
    const alphaProdT =
      1.0 - timestep / this.numTrainTimesteps; /* v8 ignore next */ /* v8 ignore next */
    const betaProdT = 1 - alphaProdT; /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < sample.length; i++) {
      /* v8 ignore next */ /* v8 ignore next */
      out[i] =
        (sample[i]! - Math.sqrt(betaProdT) * modelOutput[i]!) /
        Math.sqrt(alphaProdT); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return out; /* v8 ignore next */ /* v8 ignore next */
  }
}

export class DDPMScheduler extends Scheduler {
  /* v8 ignore next */ /* v8 ignore next */
  override step(
    /* v8 ignore next */ /* v8 ignore next */
    modelOutput: Float32Array | number[] /* v8 ignore next */ /* v8 ignore next */,
    timestep: number /* v8 ignore next */ /* v8 ignore next */,
    sample: Float32Array | number[] /* v8 ignore next */ /* v8 ignore next */,
    gen?: PyTorchPCG /* v8 ignore next */ /* v8 ignore next */,
  ): Float32Array | number[] {
    /* v8 ignore next */ /* v8 ignore next */
    const isArray = Array.isArray(sample); /* v8 ignore next */ /* v8 ignore next */
    const out = isArray
      ? new Array(sample.length)
      : new Float32Array(sample.length); /* v8 ignore next */ /* v8 ignore next */
    const alphaT =
      1.0 - timestep / this.numTrainTimesteps; /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < sample.length; i++) {
      /* v8 ignore next */ /* v8 ignore next */
      out[i] =
        (sample[i]! - (1 - alphaT) * modelOutput[i]!) /
        Math.sqrt(alphaT); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return out; /* v8 ignore next */ /* v8 ignore next */
  }
}

export class EulerDiscreteScheduler extends Scheduler {
  /* v8 ignore next */ /* v8 ignore next */
  override step(
    /* v8 ignore next */ /* v8 ignore next */
    modelOutput: Float32Array | number[] /* v8 ignore next */ /* v8 ignore next */,
    timestep: number /* v8 ignore next */ /* v8 ignore next */,
    sample: Float32Array | number[] /* v8 ignore next */ /* v8 ignore next */,
    gen?: PyTorchPCG /* v8 ignore next */ /* v8 ignore next */,
  ): Float32Array | number[] {
    /* v8 ignore next */ /* v8 ignore next */
    const isArray = Array.isArray(sample); /* v8 ignore next */ /* v8 ignore next */
    const out = isArray
      ? new Array(sample.length)
      : new Float32Array(sample.length); /* v8 ignore next */ /* v8 ignore next */
    const sigma = timestep / this.numTrainTimesteps; /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < sample.length; i++) {
      /* v8 ignore next */ /* v8 ignore next */
      out[i] = sample[i]! + modelOutput[i]! * sigma; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return out; /* v8 ignore next */ /* v8 ignore next */
  }
}

export class LCMScheduler extends Scheduler {
  /* v8 ignore next */ /* v8 ignore next */
  override step(
    /* v8 ignore next */ /* v8 ignore next */
    modelOutput: Float32Array | number[] /* v8 ignore next */ /* v8 ignore next */,
    timestep: number /* v8 ignore next */ /* v8 ignore next */,
    sample: Float32Array | number[] /* v8 ignore next */ /* v8 ignore next */,
    gen?: PyTorchPCG /* v8 ignore next */ /* v8 ignore next */,
  ): Float32Array | number[] {
    /* v8 ignore next */ /* v8 ignore next */
    const isArray = Array.isArray(sample); /* v8 ignore next */ /* v8 ignore next */
    const out = isArray
      ? new Array(sample.length)
      : new Float32Array(sample.length); /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < sample.length; i++) {
      /* v8 ignore next */ /* v8 ignore next */
      out[i] = sample[i]! - modelOutput[i]!; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return out; /* v8 ignore next */ /* v8 ignore next */
  }
}

export class PNDMScheduler extends Scheduler {}
export class LMSDiscreteScheduler extends Scheduler {}
export class DPMSolverMultistepScheduler extends Scheduler {}
export class DPMSolverSinglestepScheduler extends Scheduler {}
export class KDPM2DiscreteScheduler extends Scheduler {}
export class KDPM2AncestralDiscreteScheduler extends Scheduler {}
export class HeunDiscreteScheduler extends Scheduler {}
export class UniPCMultistepScheduler extends Scheduler {}
export class EulerAncestralDiscreteScheduler extends Scheduler {}
