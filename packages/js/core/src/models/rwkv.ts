/**
 * @fileoverview rwkv.ts
 * Provides rwkv functionality for the core package.
 */
import { Tensor } from '../ir/tensor.js';
import { recordOp } from '../macros.js';
import { Gemm, LayerNormalization } from '../primitives.js';

/**
 * Executes `getParam`.
 * @param name Parameter `name`.
 * @param shape Parameter `shape`.
 * @param dtype Parameter `dtype`.
 * @returns Result of `getParam`.
 */
function getParam(
  name: string,
  shape: number[],
  dtype: ReturnType<typeof JSON.parse> = 'float32',
): Tensor {
  return new Tensor(name, shape, dtype, false, false, new Float32Array());
}

/** Represents the `RNN` class. */
export class RNN {
  /** Property `hiddenSize` of `RNN`. */
  public hiddenSize: number;
  /** Property `direction` of `RNN`. */
  public direction: string;

  /**
   * Initializes a new `RNN` instance.
   * @param hiddenSize Parameter `hiddenSize`.
   * @param direction Parameter `direction`.
   */
  constructor(hiddenSize: number, direction: string = 'forward') {
    this.hiddenSize = hiddenSize;
    this.direction = direction;
  }

  /**
   * Executes `call` on `RNN`.
   * @param x Parameter `x`.
   * @param w Parameter `w`.
   * @param r Parameter `r`.
   * @returns Result of `call`.
   */
  call(x: Tensor, w: Tensor, r: Tensor): Tensor {
    return recordOp('RNN', [x, w, r], {
      direction: this.direction,
      hidden_size: this.hiddenSize,
    });
  }
}

/** Represents the `RWKVTimeMix` class. */
export class RWKVTimeMix {
  /** Property `prefix` of `RWKVTimeMix`. */
  public prefix: string;
  /** Property `dim` of `RWKVTimeMix`. */
  public dim: number;
  /** Property `rnn` of `RWKVTimeMix`. */
  public rnn: RNN;
  /** Property `key` of `RWKVTimeMix`. */
  public key: Gemm;
  /** Property `value` of `RWKVTimeMix`. */
  public value: Gemm;
  /** Property `receptance` of `RWKVTimeMix`. */
  public receptance: Gemm;
  /** Property `output` of `RWKVTimeMix`. */
  public output: Gemm;

  /**
   * Initializes a new `RWKVTimeMix` instance.
   * @param dim Parameter `dim`.
   * @param prefix Parameter `prefix`.
   */
  constructor(dim: number, prefix: string = '') {
    this.prefix = prefix;
    this.dim = dim;
    this.rnn = new RNN(dim);
    this.key = new Gemm(1.0, 1.0, 0, 1);
    this.value = new Gemm(1.0, 1.0, 0, 1);
    this.receptance = new Gemm(1.0, 1.0, 0, 1);
    this.output = new Gemm(1.0, 1.0, 0, 1);
  }

  /**
   * Executes `call` on `RWKVTimeMix`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    const xT = recordOp('Transpose', [x], { perm: [1, 0, 2] });

    const w = getParam(`${this.prefix}.rnn.w`, [1, this.dim, this.dim]);
    const r = getParam(`${this.prefix}.rnn.r`, [1, this.dim, this.dim]);

    let rnnOut = this.rnn.call(xT, w, r);
    rnnOut = recordOp('Transpose', [rnnOut], { perm: [1, 0, 2] });

    const k = this.key.call(rnnOut, getParam(`${this.prefix}.key.weight`, [this.dim, this.dim]));
    const v = this.value.call(
      rnnOut,
      getParam(`${this.prefix}.value.weight`, [this.dim, this.dim]),
    );
    const rec = this.receptance.call(
      x,
      getParam(`${this.prefix}.receptance.weight`, [this.dim, this.dim]),
    );

    const kv = recordOp('Mul', [k, v]);
    let out = recordOp('Mul', [recordOp('Sigmoid', [rec]), kv]);
    out = this.output.call(out, getParam(`${this.prefix}.output.weight`, [this.dim, this.dim]));

    return out;
  }
}

/** Represents the `RWKVChannelMix` class. */
export class RWKVChannelMix {
  /** Property `prefix` of `RWKVChannelMix`. */
  public prefix: string;
  /** Property `dim` of `RWKVChannelMix`. */
  public dim: number;
  /** Property `key` of `RWKVChannelMix`. */
  public key: Gemm;
  /** Property `receptance` of `RWKVChannelMix`. */
  public receptance: Gemm;
  /** Property `value` of `RWKVChannelMix`. */
  public value: Gemm;

  /**
   * Initializes a new `RWKVChannelMix` instance.
   * @param dim Parameter `dim`.
   * @param prefix Parameter `prefix`.
   */
  constructor(dim: number, prefix: string = '') {
    this.prefix = prefix;
    this.dim = dim;
    this.key = new Gemm(1.0, 1.0, 0, 1);
    this.receptance = new Gemm(1.0, 1.0, 0, 1);
    this.value = new Gemm(1.0, 1.0, 0, 1);
  }

  /**
   * Executes `call` on `RWKVChannelMix`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    let k = this.key.call(x, getParam(`${this.prefix}.key.weight`, [this.dim * 4, this.dim]));
    k = recordOp('Relu', [k]);

    const v = this.value.call(k, getParam(`${this.prefix}.value.weight`, [this.dim, this.dim * 4]));

    const rec = this.receptance.call(
      x,
      getParam(`${this.prefix}.receptance.weight`, [this.dim, this.dim]),
    );
    const out = recordOp('Mul', [recordOp('Sigmoid', [rec]), v]);

    return out;
  }
}

/** Represents the `RWKVBlock` class. */
export class RWKVBlock {
  /** Property `prefix` of `RWKVBlock`. */
  public prefix: string;
  /** Property `dim` of `RWKVBlock`. */
  public dim: number;
  /** Property `norm1` of `RWKVBlock`. */
  public norm1: LayerNormalization;
  /** Property `timeMix` of `RWKVBlock`. */
  public timeMix: RWKVTimeMix;
  /** Property `norm2` of `RWKVBlock`. */
  public norm2: LayerNormalization;
  /** Property `channelMix` of `RWKVBlock`. */
  public channelMix: RWKVChannelMix;

  /**
   * Initializes a new `RWKVBlock` instance.
   * @param dim Parameter `dim`.
   * @param prefix Parameter `prefix`.
   */
  constructor(dim: number, prefix: string = '') {
    this.prefix = prefix;
    this.dim = dim;
    this.norm1 = new LayerNormalization([dim]);
    this.timeMix = new RWKVTimeMix(dim, `${prefix}.att`);
    this.norm2 = new LayerNormalization([dim]);
    this.channelMix = new RWKVChannelMix(dim, `${prefix}.ffn`);
  }

  /**
   * Executes `call` on `RWKVBlock`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    let identity = x;
    let xNorm = this.norm1.call(
      x,
      getParam(`${this.prefix}.norm1.weight`, [this.dim]),
      getParam(`${this.prefix}.norm1.bias`, [this.dim]),
    );
    const xAtt = this.timeMix.call(xNorm);
    x = recordOp('Add', [identity, xAtt]);

    identity = x;
    xNorm = this.norm2.call(
      x,
      getParam(`${this.prefix}.norm2.weight`, [this.dim]),
      getParam(`${this.prefix}.norm2.bias`, [this.dim]),
    );
    const xFfn = this.channelMix.call(xNorm);
    x = recordOp('Add', [identity, xFfn]);

    return x;
  }
}

/** Represents the `RWKV` class. */
export class RWKV {
  /** Property `vocabSize` of `RWKV`. */
  public vocabSize: number;
  /** Property `dim` of `RWKV`. */
  public dim: number;
  /** Property `blocks` of `RWKV`. */
  public blocks: RWKVBlock[];
  /** Property `norm` of `RWKV`. */
  public norm: LayerNormalization;
  /** Property `head` of `RWKV`. */
  public head: Gemm;

  /**
   * Initializes a new `RWKV` instance.
   * @param vocabSize Parameter `vocabSize`.
   * @param dim Parameter `dim`.
   * @param depth Parameter `depth`.
   */
  constructor(vocabSize: number = 50277, dim: number = 768, depth: number = 24) {
    this.vocabSize = vocabSize;
    this.dim = dim;

    this.blocks = [];
    for (let i = 0; i < depth; i++) {
      this.blocks.push(new RWKVBlock(dim, `blocks.${i}`));
    }

    this.norm = new LayerNormalization([dim]);
    this.head = new Gemm(1.0, 1.0, 0, 1);
  }

  /**
   * Executes `call` on `RWKV`.
   * @param inputIds Parameter `inputIds`.
   * @returns Result of `call`.
   */
  call(inputIds: Tensor): Tensor {
    let x = recordOp(
      'Gather',
      [getParam('embedding.weight', [this.vocabSize, this.dim]), inputIds],
      { axis: 0 },
    );

    for (const block of this.blocks) {
      x = block.call(x);
    }

    x = this.norm.call(x, getParam('norm.weight', [this.dim]), getParam('norm.bias', [this.dim]));
    x = this.head.call(x, getParam('head.weight', [this.vocabSize, this.dim]));

    return x;
  }
}

/**
 * Executes `rwkvV4`.
 * @returns Result of `rwkvV4`.
 */
export function rwkvV4(): RWKV {
  return new RWKV(50277, 768, 24);
}
