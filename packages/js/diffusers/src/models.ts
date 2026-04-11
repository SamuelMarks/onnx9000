/* eslint-disable */
export class AutoencoderKL {
  encode(x: Float32Array): Float32Array {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i]! * 0.18215;
    return out;
  }
  decode(x: Float32Array): Float32Array {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i]! / 0.18215;
    return out;
  }
}

export class UNet2DConditionModel {
  call(sample: Float32Array, timestep: number, encoder_hidden_states: Float32Array): Float32Array {
    const out = new Float32Array(sample.length);
    for (let i = 0; i < sample.length; i++) {
      out[i] = sample[i]! - timestep * 0.01;
    }
    return out;
  }
}
