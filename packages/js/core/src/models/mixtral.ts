/**
 * @fileoverview mixtral.ts
 * Provides mixtral functionality for the core package.
 */
import { Tensor } from '../ir/tensor.js';
import { recordOp } from '../macros.js';
import { Gemm, GroupedQueryAttention, RMSNorm, RoPE } from '../primitives.js';
import { SwiGLU } from './llama.js';

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

/** Represents the `SparseMoE` class. */
export class SparseMoE {
  /** Property `prefix` of `SparseMoE`. */
  public prefix: string;
  /** Property `numExperts` of `SparseMoE`. */
  public numExperts: number;
  /** Property `topK` of `SparseMoE`. */
  public topK: number;
  /** Property `dim` of `SparseMoE`. */
  public dim: number;
  /** Property `ffnDim` of `SparseMoE`. */
  public ffnDim: number;
  /** Property `gate` of `SparseMoE`. */
  public gate: Gemm;
  /** Property `experts` of `SparseMoE`. */
  public experts: SwiGLU[];

  /**
   * Initializes a new `SparseMoE` instance.
   * @param numExperts Parameter `numExperts`.
   * @param topK Parameter `topK`.
   * @param dim Parameter `dim`.
   * @param ffnDim Parameter `ffnDim`.
   * @param prefix Parameter `prefix`.
   */
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

  /**
   * Executes `call` on `SparseMoE`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    const logits = this.gate.call(
      x,
      getParam(`${this.prefix}.gate.weight`, [this.numExperts, this.dim]),
    );
    const kTensor = recordOp('Constant', [], { value: [this.topK], dtype: 7 });
    const scores = recordOp('Softmax', [logits], { axis: -1 });
    const topkOut = recordOp('TopK', [scores, kTensor], { axis: -1 });

    const gathered = recordOp('GatherND', [x, topkOut]);
    const expertOut = this.experts[0]?.call(gathered) ?? gathered;
    const out = recordOp('ScatterND', [topkOut, expertOut, x]);

    return out;
  }
}

/** Represents the `MixtralBlock` class. */
export class MixtralBlock {
  /** Property `prefix` of `MixtralBlock`. */
  public prefix: string;
  /** Property `dim` of `MixtralBlock`. */
  public dim: number;
  /** Property `norm1` of `MixtralBlock`. */
  public norm1: RMSNorm;
  /** Property `attn` of `MixtralBlock`. */
  public attn: GroupedQueryAttention;
  /** Property `norm2` of `MixtralBlock`. */
  public norm2: RMSNorm;
  /** Property `moe` of `MixtralBlock`. */
  public moe: SparseMoE;

  /**
   * Initializes a new `MixtralBlock` instance.
   * @param dim Parameter `dim`.
   * @param numHeads Parameter `numHeads`.
   * @param numKvHeads Parameter `numKvHeads`.
   * @param ffnDim Parameter `ffnDim`.
   * @param numExperts Parameter `numExperts`.
   * @param topK Parameter `topK`.
   * @param prefix Parameter `prefix`.
   */
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

  /**
   * Executes `call` on `MixtralBlock`.
   * @param x Parameter `x`.
   * @param _pos Parameter `_pos`.
   * @param mask Parameter `mask`.
   * @returns Result of `call`.
   */
  call(x: Tensor, _pos: Tensor, mask?: Tensor): Tensor {
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

/** Represents the `Mixtral` class. */
export class Mixtral {
  /** Property `vocabSize` of `Mixtral`. */
  public vocabSize: number;
  /** Property `dim` of `Mixtral`. */
  public dim: number;
  /** Property `depth` of `Mixtral`. */
  public depth: number;
  /** Property `maxSeqLen` of `Mixtral`. */
  public maxSeqLen: number;
  /** Property `blocks` of `Mixtral`. */
  public blocks: MixtralBlock[];
  /** Property `norm` of `Mixtral`. */
  public norm: RMSNorm;
  /** Property `lmHead` of `Mixtral`. */
  public lmHead: Gemm;
  /** Property `rope` of `Mixtral`. */
  public rope: RoPE;

  /**
   * Initializes a new `Mixtral` instance.
   * @param vocabSize Parameter `vocabSize`.
   * @param dim Parameter `dim`.
   * @param numHeads Parameter `numHeads`.
   * @param numKvHeads Parameter `numKvHeads`.
   * @param depth Parameter `depth`.
   * @param ffnDim Parameter `ffnDim`.
   * @param numExperts Parameter `numExperts`.
   * @param topK Parameter `topK`.
   * @param maxSeqLen Parameter `maxSeqLen`.
   */
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

  /**
   * Executes `call` on `Mixtral`.
   * @param inputIds Parameter `inputIds`.
   * @param pos Parameter `pos`.
   * @param mask Parameter `mask`.
   * @returns Result of `call`.
   */
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

/**
 * Executes `mixtral8x7b`.
 * @returns Result of `mixtral8x7b`.
 */
export function mixtral8x7b(): Mixtral {
  return new Mixtral(32000, 4096, 32, 8, 32, 14336, 8, 2);
}
