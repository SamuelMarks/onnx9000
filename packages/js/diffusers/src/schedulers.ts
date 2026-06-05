import { PyTorchPCG } from './utils';

export class Scheduler {
  numTrainTimesteps: number;
  timesteps: number[];

  constructor(numTrainTimesteps: number = 1000) {
    this.numTrainTimesteps = numTrainTimesteps;
    this.timesteps = Array.from({ length: numTrainTimesteps }, (_, i) => numTrainTimesteps - 1 - i);
  }

  setTimesteps(numInferenceSteps: number): void {
    const step = Math.floor(this.numTrainTimesteps / numInferenceSteps);
    this.timesteps = Array.from(
      { length: numInferenceSteps },
      (_, i) => this.numTrainTimesteps - 1 - i * step,
    );
  }

  step(
    modelOutput: Float32Array | number[],
    timestep: number,
    sample: Float32Array | number[],
    gen?: PyTorchPCG,
  ): Float32Array | number[] {
    return sample;
  }
}

export class DDIMScheduler extends Scheduler {
  override step(
    modelOutput: Float32Array | number[],
    timestep: number,
    sample: Float32Array | number[],
    gen?: PyTorchPCG,
  ): Float32Array | number[] {
    const isArray = Array.isArray(sample);
    const out = isArray ? new Array(sample.length) : new Float32Array(sample.length);
    const alphaProdT = 1.0 - timestep / this.numTrainTimesteps;
    const betaProdT = 1 - alphaProdT;
    for (let i = 0; i < sample.length; i++) {
      out[i] = (sample[i]! - Math.sqrt(betaProdT) * modelOutput[i]!) / Math.sqrt(alphaProdT);
    }
    return out;
  }
}

export class DDPMScheduler extends Scheduler {
  override step(
    modelOutput: Float32Array | number[],
    timestep: number,
    sample: Float32Array | number[],
    gen?: PyTorchPCG,
  ): Float32Array | number[] {
    const isArray = Array.isArray(sample);
    const out = isArray ? new Array(sample.length) : new Float32Array(sample.length);
    const alphaT = 1.0 - timestep / this.numTrainTimesteps;
    for (let i = 0; i < sample.length; i++) {
      out[i] = (sample[i]! - (1 - alphaT) * modelOutput[i]!) / Math.sqrt(alphaT);
    }
    return out;
  }
}

export class EulerDiscreteScheduler extends Scheduler {
  override step(
    modelOutput: Float32Array | number[],
    timestep: number,
    sample: Float32Array | number[],
    gen?: PyTorchPCG,
  ): Float32Array | number[] {
    const isArray = Array.isArray(sample);
    const out = isArray ? new Array(sample.length) : new Float32Array(sample.length);
    const sigma = timestep / this.numTrainTimesteps;
    for (let i = 0; i < sample.length; i++) {
      out[i] = sample[i]! + modelOutput[i]! * sigma;
    }
    return out;
  }
}

export class LCMScheduler extends Scheduler {
  override step(
    modelOutput: Float32Array | number[],
    timestep: number,
    sample: Float32Array | number[],
    gen?: PyTorchPCG,
  ): Float32Array | number[] {
    const isArray = Array.isArray(sample);
    const out = isArray ? new Array(sample.length) : new Float32Array(sample.length);
    for (let i = 0; i < sample.length; i++) {
      out[i] = sample[i]! - modelOutput[i]!;
    }
    return out;
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
