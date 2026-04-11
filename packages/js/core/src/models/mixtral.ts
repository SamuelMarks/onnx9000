/* eslint-disable */
import { Tensor } from '../ir/tensor.js';
import { Gemm, GroupedQueryAttention, RMSNorm, RoPE } from '../primitives.js';
import { SwiGLU } from './llama.js';

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

export class SparseMoE {
  public prefix: string;
  public numExperts: number;
  public topK: number;
  public dim: number;
  public ffnDim: number;
  public gate: Gemm;
  public experts: SwiGLU[];

  constructor(numExperts: number, topK: number, dim: number, ffnDim: number, prefix: string = '') {
    this.prefix = prefix;
    this.numExperts = numExperts;
    this.topK = topK;
    this.dim = dim;
    this.ffnDim = ffnDim;

    this.gate = new Gemm(1.0, 1.0, 0, 1);
    this.experts = [];
    for (let i = 0; i < numExperts; i++) {
      this.experts.push(new SwiGLU(dim, ffnDim, `${prefix}.experts.${i}`));
    }
  }

  call(x: Tensor): Tensor {
    const logits = this.gate.call(
      x,
      getParam(`${this.prefix}.gate.weight`, [this.numExperts, this.dim]),
    );
    const kTensor = recordOp('Constant', [], { value: [this.topK], dtype: 7 });
    const scores = recordOp('Softmax', [logits], { axis: -1 });
    const topkOut = recordOp('TopK', [scores, kTensor], { axis: -1 });

    const gathered = recordOp('GatherND', [x, topkOut]);
    const expertOut = this.experts[0]!.call(gathered);
    const out = recordOp('ScatterND', [topkOut, expertOut, x]);

    return out;
  }
}

export class MixtralBlock {
  public prefix: string;
  public dim: number;
  public norm1: RMSNorm;
  public attn: GroupedQueryAttention;
  public norm2: RMSNorm;
  public moe: SparseMoE;

  constructor(
    dim: number,
    numHeads: number,
    numKvHeads: number,
    ffnDim: number,
    numExperts: number,
    topK: number,
    prefix: string = '',
  ) {
    this.prefix = prefix;
    this.dim = dim;
    this.norm1 = new RMSNorm([dim]);
    this.attn = new GroupedQueryAttention(numHeads, numKvHeads, false, false);
    this.norm2 = new RMSNorm([dim]);
    this.moe = new SparseMoE(numExperts, topK, dim, ffnDim, `${prefix}.moe`);
  }

  call(x: Tensor, pos: Tensor, mask?: Tensor): Tensor {
    let identity = x;
    let xNorm = this.norm1.call(x, getParam(`${this.prefix}.norm1.weight`, [this.dim]));
    const xAttn = this.attn.call(xNorm, xNorm, xNorm, mask);
    x = recordOp('Add', [identity, xAttn]);

    identity = x;
    xNorm = this.norm2.call(x, getParam(`${this.prefix}.norm2.weight`, [this.dim]));
    const xMoe = this.moe.call(xNorm);
    x = recordOp('Add', [identity, xMoe]);

    return x;
  }
}

export class Mixtral {
  public vocabSize: number;
  public dim: number;
  public depth: number;
  public maxSeqLen: number;
  public blocks: MixtralBlock[];
  public norm: RMSNorm;
  public lmHead: Gemm;
  public rope: RoPE;

  constructor(
    vocabSize: number = 32000,
    dim: number = 4096,
    numHeads: number = 32,
    numKvHeads: number = 8,
    depth: number = 32,
    ffnDim: number = 14336,
    numExperts: number = 8,
    topK: number = 2,
    maxSeqLen: number = 4096,
  ) {
    this.vocabSize = vocabSize;
    this.dim = dim;
    this.depth = depth;
    this.maxSeqLen = maxSeqLen;

    this.blocks = [];
    for (let i = 0; i < depth; i++) {
      this.blocks.push(
        new MixtralBlock(dim, numHeads, numKvHeads, ffnDim, numExperts, topK, `blocks.${i}`),
      );
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

export function mixtral8x7b(): Mixtral {
  return new Mixtral(32000, 4096, 32, 8, 32, 14336, 8, 2);
}
