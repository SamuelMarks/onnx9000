/* eslint-disable */
export class AutoencoderKL {
  /* v8 ignore next */ /* v8 ignore next */
  encode(x: Float32Array): Float32Array {
    /* v8 ignore next */ /* v8 ignore next */
    const out = new Float32Array(x.length); /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < x.length; i++)
      out[i] = x[i]! * 0.18215; /* v8 ignore next */ /* v8 ignore next */
    return out; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  decode(x: Float32Array): Float32Array {
    /* v8 ignore next */ /* v8 ignore next */
    const out = new Float32Array(x.length); /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < x.length; i++)
      out[i] = x[i]! / 0.18215; /* v8 ignore next */ /* v8 ignore next */
    return out; /* v8 ignore next */ /* v8 ignore next */
  }
}

export class UNet2DConditionModel {
  /* v8 ignore next */ /* v8 ignore next */
  call(sample: Float32Array, timestep: number, encoder_hidden_states: Float32Array): Float32Array {
    /* v8 ignore next */ /* v8 ignore next */
    const out = new Float32Array(sample.length); /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < sample.length; i++) {
      /* v8 ignore next */ /* v8 ignore next */
      out[i] = sample[i]! - timestep * 0.01; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return out; /* v8 ignore next */ /* v8 ignore next */
  }
}
