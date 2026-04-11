/* eslint-disable */
import { Tensor } from '../ir/tensor.js';
import { Gemm, GroupedQueryAttention, RMSNorm, RoPE, Silu } from '../primitives.js';

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

export class SwiGLU {
  public prefix: string;
  public hiddenDim: number;
  public ffnDim: number;
  public w1: Gemm;
  public w2: Gemm;
  public w3: Gemm;
  public act: Silu;

  constructor(hiddenDim: number, ffnDim: number, prefix: string = '') {
    this.prefix = prefix;
    this.hiddenDim = hiddenDim;
    this.ffnDim = ffnDim;
    this.w1 = new Gemm(1.0, 1.0, 0, 1);
    this.w2 = new Gemm(1.0, 1.0, 0, 1);
    this.w3 = new Gemm(1.0, 1.0, 0, 1);
    this.act = new Silu();
  }

  call(x: Tensor): Tensor {
    const gate = this.w1.call(
      x,
      getParam(`${this.prefix}.w1.weight`, [this.ffnDim, this.hiddenDim]),
    );
    const up = this.w3.call(x, getParam(`${this.prefix}.w3.weight`, [this.ffnDim, this.hiddenDim]));

    const activatedGate = this.act.call(gate);
    const hidden = recordOp('Mul', [activatedGate, up]);

    const down = this.w2.call(
      hidden,
      getParam(`${this.prefix}.w2.weight`, [this.hiddenDim, this.ffnDim]),
    );
    return down;
  }
}

export class LLaMABlock {
  public prefix: string;
  public dim: number;
  public norm1: RMSNorm;
  public attn: GroupedQueryAttention;
  public norm2: RMSNorm;
  public mlp: SwiGLU;

  constructor(
    dim: number,
    numHeads: number,
    numKvHeads: number,
    ffnDim: number,
    prefix: string = '',
  ) {
    this.prefix = prefix;
    this.dim = dim;
    this.norm1 = new RMSNorm([dim]);
    this.attn = new GroupedQueryAttention(numHeads, numKvHeads, false, false);
    this.norm2 = new RMSNorm([dim]);
    this.mlp = new SwiGLU(dim, ffnDim, `${prefix}.mlp`);
  }

  call(x: Tensor, pos: Tensor, mask?: Tensor): Tensor {
    let identity = x;
    let xNorm = this.norm1.call(x, getParam(`${this.prefix}.norm1.weight`, [this.dim]));
    const xAttn = this.attn.call(xNorm, xNorm, xNorm, mask);
    x = recordOp('Add', [identity, xAttn]);

    identity = x;
    xNorm = this.norm2.call(x, getParam(`${this.prefix}.norm2.weight`, [this.dim]));
    const xMlp = this.mlp.call(xNorm);
    x = recordOp('Add', [identity, xMlp]);

    return x;
  }
}

export class LLaMA {
  public vocabSize: number;
  public dim: number;
  public depth: number;
  public maxSeqLen: number;
  public blocks: LLaMABlock[];
  public norm: RMSNorm;
  public lmHead: Gemm;
  public rope: RoPE;

  constructor(
    vocabSize: number = 32000,
    dim: number = 4096,
    numHeads: number = 32,
    numKvHeads: number = 32,
    depth: number = 32,
    ffnDim: number = 11008,
    maxSeqLen: number = 2048,
  ) {
    this.vocabSize = vocabSize;
    this.dim = dim;
    this.depth = depth;
    this.maxSeqLen = maxSeqLen;

    this.blocks = [];
    for (let i = 0; i < depth; i++) {
      this.blocks.push(new LLaMABlock(dim, numHeads, numKvHeads, ffnDim, `blocks.${i}`));
    }
    this.norm = new RMSNorm([dim]);
    this.lmHead = new Gemm(1.0, 1.0, 0, 1);
    this.rope = new RoPE(Math.floor(dim / numHeads), 10000.0, maxSeqLen);
  }

  call(inputIds: Tensor, pos: Tensor, mask?: Tensor): Tensor {
    let x = recordOp(
      'Gather',
      [getParam('tok_embeddings.weight', [this.vocabSize, this.dim]), inputIds],
      { axis: 0 },
    );
    x = this.rope.call(x, pos);

    for (const block of this.blocks) {
      x = block.call(x, pos, mask);
    }

    x = this.norm.call(x, getParam('norm.weight', [this.dim]));
    x = this.lmHead.call(x, getParam('output.weight', [this.vocabSize, this.dim]));
    return x;
  }
}

export function llama7b(): LLaMA {
  return new LLaMA(32000, 4096, 32, 32, 32, 11008);
}
