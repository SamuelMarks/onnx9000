/**
 * @fileoverview models.ts
 * Provides models functionality for the diffusers package.
 */
export class AutoencoderKL {
  /**
   * Executes `encode` on `AutoencoderKL`.
   * @param x Parameter `x`.
   * @returns Result of `encode`.
   */
  encode(x: Float32Array): Float32Array {
    const out = new Float32Array(x.length);
    for (const [i, v] of x.entries()) out[i] = v * 0.18215;
    return out;
  }
  /**
   * Executes `decode` on `AutoencoderKL`.
   * @param x Parameter `x`.
   * @returns Result of `decode`.
   */
  decode(x: Float32Array): Float32Array {
    const out = new Float32Array(x.length);
    for (const [i, v] of x.entries()) out[i] = v / 0.18215;
    return out;
  }
}

/** Represents the `UNet2DConditionModel` class. */
export class UNet2DConditionModel {
  /**
   * Executes `call` on `UNet2DConditionModel`.
   * @param sample Parameter `sample`.
   * @param timestep Parameter `timestep`.
   * @param _encoder_hidden_states Parameter `_encoder_hidden_states`.
   * @returns Result of `call`.
   */
  call(sample: Float32Array, timestep: number, _encoder_hidden_states: Float32Array): Float32Array {
    const out = new Float32Array(sample.length);
    for (const [i, v] of sample.entries()) {
      out[i] = v - timestep * 0.01;
    }
    return out;
  }
}
