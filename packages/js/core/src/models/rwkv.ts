import { Tensor } from '../ir/tensor.js';
import { Gemm, LayerNormalization } from '../primitives.js';

function getParam(
  name: string,
  shape: number[],
  dtype: ReturnType<typeof JSON.parse> = 'float32',
): Tensor {
  return new Tensor(name, shape, dtype, false, false, new Float32Array());
}

function recordOp(opType: string, inputs: Tensor[], attr?: ReturnType<typeof JSON.parse>): Tensor {
  const dtype = inputs[0]?.dtype ?? 'float32';
  return new Tensor(`${opType}_out`, [], dtype, false, false, new Float32Array());
}

export class RNN {
  public hiddenSize: number;
  public direction: string;

  constructor(hiddenSize: number, direction: string = 'forward') {
    this.hiddenSize = hiddenSize;
    this.direction = direction;
  }

  call(x: Tensor, w: Tensor, r: Tensor): Tensor {
    return recordOp('RNN', [x, w, r], { direction: this.direction, hidden_size: this.hiddenSize });
  }
}

export class RWKVTimeMix {
  public prefix: string;
  public dim: number;
  public rnn: RNN;
  public key: Gemm;
  public value: Gemm;
  public receptance: Gemm;
  public output: Gemm;

  constructor(dim: number, prefix: string = '') {
    this.prefix = prefix;
    this.dim = dim;
    this.rnn = new RNN(dim);
    this.key = new Gemm(1.0, 1.0, 0, 1);
    this.value = new Gemm(1.0, 1.0, 0, 1);
    this.receptance = new Gemm(1.0, 1.0, 0, 1);
    this.output = new Gemm(1.0, 1.0, 0, 1);
  }

  call(x: Tensor): Tensor {
    let xT = recordOp('Transpose', [x], { perm: [1, 0, 2] });

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

export class RWKVChannelMix {
  public prefix: string;
  public dim: number;
  public key: Gemm;
  public receptance: Gemm;
  public value: Gemm;

  constructor(dim: number, prefix: string = '') {
    this.prefix = prefix;
    this.dim = dim;
    this.key = new Gemm(1.0, 1.0, 0, 1);
    this.receptance = new Gemm(1.0, 1.0, 0, 1);
    this.value = new Gemm(1.0, 1.0, 0, 1);
  }

  call(x: Tensor): Tensor {
    let k = this.key.call(x, getParam(`${this.prefix}.key.weight`, [this.dim * 4, this.dim]));
    k = recordOp('Relu', [k]);

    let v = this.value.call(k, getParam(`${this.prefix}.value.weight`, [this.dim, this.dim * 4]));

    const rec = this.receptance.call(
      x,
      getParam(`${this.prefix}.receptance.weight`, [this.dim, this.dim]),
    );
    const out = recordOp('Mul', [recordOp('Sigmoid', [rec]), v]);

    return out;
  }
}

export class RWKVBlock {
  public prefix: string;
  public dim: number;
  public norm1: LayerNormalization;
  public timeMix: RWKVTimeMix;
  public norm2: LayerNormalization;
  public channelMix: RWKVChannelMix;

  constructor(dim: number, prefix: string = '') {
    this.prefix = prefix;
    this.dim = dim;
    this.norm1 = new LayerNormalization([dim]);
    this.timeMix = new RWKVTimeMix(dim, `${prefix}.att`);
    this.norm2 = new LayerNormalization([dim]);
    this.channelMix = new RWKVChannelMix(dim, `${prefix}.ffn`);
  }

  call(x: Tensor): Tensor {
    let identity = x;
    let xNorm = this.norm1.call(
      x,
      getParam(`${this.prefix}.norm1.weight`, [this.dim]),
      getParam(`${this.prefix}.norm1.bias`, [this.dim]),
    );
    let xAtt = this.timeMix.call(xNorm);
    x = recordOp('Add', [identity, xAtt]);

    identity = x;
    xNorm = this.norm2.call(
      x,
      getParam(`${this.prefix}.norm2.weight`, [this.dim]),
      getParam(`${this.prefix}.norm2.bias`, [this.dim]),
    );
    let xFfn = this.channelMix.call(xNorm);
    x = recordOp('Add', [identity, xFfn]);

    return x;
  }
}

export class RWKV {
  public vocabSize: number;
  public dim: number;
  public blocks: RWKVBlock[];
  public norm: LayerNormalization;
  public head: Gemm;

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

export function rwkvV4(): RWKV {
  return new RWKV(50277, 768, 24);
}
