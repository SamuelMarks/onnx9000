/**
 * @fileoverview mae.ts
 * Provides mae functionality for the core package.
 */
import { Tensor } from '../ir/tensor.js';
import { Gemm, LayerNormalization } from '../primitives.js';
import { Block, PatchEmbed } from './vit.js';

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

/**
 * Executes `recordOp`.
 * @param opType Parameter `opType`.
 * @param inputs Parameter `inputs`.
 * @param _attr Parameter `_attr`.
 * @returns Result of `recordOp`.
 */
function recordOp(opType: string, inputs: Tensor[], _attr?: ReturnType<typeof JSON.parse>): Tensor {
  const dtype = inputs[0]?.dtype ?? 'float32';
  return new Tensor(`${opType}_out`, [], dtype, false, false, new Float32Array());
}

/** Represents the `MaskedAutoencoderViT` class. */
export class MaskedAutoencoderViT {
  /** Property `imgSize` of `MaskedAutoencoderViT`. */
  public imgSize: number;
  /** Property `patchSize` of `MaskedAutoencoderViT`. */
  public patchSize: number;
  /** Property `numPatches` of `MaskedAutoencoderViT`. */
  public numPatches: number;
  /** Property `embedDim` of `MaskedAutoencoderViT`. */
  public embedDim: number;
  /** Property `decoderEmbedDim` of `MaskedAutoencoderViT`. */
  public decoderEmbedDim: number;

  /** Property `patchEmbed` of `MaskedAutoencoderViT`. */
  public patchEmbed: PatchEmbed;
  /** Property `blocks` of `MaskedAutoencoderViT`. */
  public blocks: Block[];
  /** Property `norm` of `MaskedAutoencoderViT`. */
  public norm: LayerNormalization;

  /** Property `decoderEmbed` of `MaskedAutoencoderViT`. */
  public decoderEmbed: Gemm;
  /** Property `decoderBlocks` of `MaskedAutoencoderViT`. */
  public decoderBlocks: Block[];
  /** Property `decoderNorm` of `MaskedAutoencoderViT`. */
  public decoderNorm: LayerNormalization;
  /** Property `decoderPred` of `MaskedAutoencoderViT`. */
  public decoderPred: Gemm;

  /**
   * Initializes a new `MaskedAutoencoderViT` instance.
   * @param imgSize Parameter `imgSize`.
   * @param patchSize Parameter `patchSize`.
   * @param inChans Parameter `inChans`.
   * @param embedDim Parameter `embedDim`.
   * @param depth Parameter `depth`.
   * @param numHeads Parameter `numHeads`.
   * @param decoderEmbedDim Parameter `decoderEmbedDim`.
   * @param decoderDepth Parameter `decoderDepth`.
   * @param decoderNumHeads Parameter `decoderNumHeads`.
   * @param mlpRatio Parameter `mlpRatio`.
   */
  constructor(
    imgSize: number = 224,
    patchSize: number = 16,
    inChans: number = 3,
    embedDim: number = 1024,
    depth: number = 24,
    numHeads: number = 16,
    decoderEmbedDim: number = 512,
    decoderDepth: number = 8,
    decoderNumHeads: number = 16,
    mlpRatio: number = 4.0,
  ) {
    this.imgSize = imgSize;
    this.patchSize = patchSize;
    this.numPatches = Math.floor(imgSize / patchSize) * Math.floor(imgSize / patchSize);

    this.embedDim = embedDim;
    this.decoderEmbedDim = decoderEmbedDim;

    this.patchEmbed = new PatchEmbed(imgSize, patchSize, inChans, embedDim, 'patch_embed');
    this.blocks = [];
    for (let i = 0; i < depth; i++) {
      this.blocks.push(new Block(embedDim, numHeads, mlpRatio, true, `blocks.${i}`));
    }
    this.norm = new LayerNormalization([embedDim]);

    this.decoderEmbed = new Gemm(1.0, 1.0, 0, 1);
    this.decoderBlocks = [];
    for (let i = 0; i < decoderDepth; i++) {
      this.decoderBlocks.push(
        new Block(decoderEmbedDim, decoderNumHeads, mlpRatio, true, `decoder_blocks.${i}`),
      );
    }
    this.decoderNorm = new LayerNormalization([decoderEmbedDim]);
    this.decoderPred = new Gemm(1.0, 1.0, 0, 1);
  }

  /**
   * Executes `forwardEncoder` on `MaskedAutoencoderViT`.
   * @param x Parameter `x`.
   * @param maskIndices Parameter `maskIndices`.
   * @returns Result of `forwardEncoder`.
   */
  forwardEncoder(x: Tensor, maskIndices: Tensor): [Tensor, Tensor] {
    x = this.patchEmbed.call(x);

    const posEmbed = getParam('pos_embed', [1, this.numPatches + 1, this.embedDim]);

    const starts = recordOp('Constant', [], { value: [1], dtype: 7 });
    const ends = recordOp('Constant', [], {
      value: [this.numPatches + 1],
      dtype: 7,
    });
    const axes = recordOp('Constant', [], { value: [1], dtype: 7 });
    const posEmbedNoCls = recordOp('Slice', [posEmbed, starts, ends, axes]);

    x = recordOp('Add', [x, posEmbedNoCls]);

    x = recordOp('Gather', [x, maskIndices], { axis: 1 });

    let clsToken = getParam('cls_token', [1, 1, this.embedDim]);
    const clsPosEmbed = recordOp('Slice', [
      posEmbed,
      recordOp('Constant', [], { value: [0], dtype: 7 }),
      recordOp('Constant', [], { value: [1], dtype: 7 }),
      recordOp('Constant', [], { value: [1], dtype: 7 }),
    ]);
    clsToken = recordOp('Add', [clsToken, clsPosEmbed]);

    x = recordOp('Concat', [clsToken, x], { axis: 1 });

    for (const blk of this.blocks) {
      x = blk.call(x);
    }
    x = this.norm.call(
      x,
      getParam('norm.weight', [this.embedDim]),
      getParam('norm.bias', [this.embedDim]),
    );

    return [x, maskIndices];
  }

  /**
   * Executes `forwardDecoder` on `MaskedAutoencoderViT`.
   * @param x Parameter `x`.
   * @param maskIndices Parameter `maskIndices`.
   * @returns Result of `forwardDecoder`.
   */
  forwardDecoder(x: Tensor, maskIndices: Tensor): Tensor {
    x = this.decoderEmbed.call(
      x,
      getParam('decoder_embed.weight', [this.decoderEmbedDim, this.embedDim]),
      getParam('decoder_embed.bias', [this.decoderEmbedDim]),
    );

    const maskToken = getParam('mask_token', [1, 1, this.decoderEmbedDim]);
    const fullMask = recordOp('Tile', [
      maskToken,
      recordOp('Constant', [], { value: [1, this.numPatches, 1], dtype: 7 }),
    ]);

    const starts = recordOp('Constant', [], { value: [1], dtype: 7 });
    const ends = recordOp('Constant', [], {
      value: [this.numPatches + 1],
      dtype: 7,
    });
    const axes = recordOp('Constant', [], { value: [1], dtype: 7 });
    const xNoCls = recordOp('Slice', [x, starts, ends, axes]);

    const xFull = recordOp('ScatterND', [fullMask, maskIndices, xNoCls]);

    const xCls = recordOp('Slice', [
      x,
      recordOp('Constant', [], { value: [0], dtype: 7 }),
      recordOp('Constant', [], { value: [1], dtype: 7 }),
      recordOp('Constant', [], { value: [1], dtype: 7 }),
    ]);
    x = recordOp('Concat', [xCls, xFull], { axis: 1 });

    const decoderPosEmbed = getParam('decoder_pos_embed', [
      1,
      this.numPatches + 1,
      this.decoderEmbedDim,
    ]);
    x = recordOp('Add', [x, decoderPosEmbed]);

    for (const blk of this.decoderBlocks) {
      x = blk.call(x);
    }
    x = this.decoderNorm.call(
      x,
      getParam('decoder_norm.weight', [this.decoderEmbedDim]),
      getParam('decoder_norm.bias', [this.decoderEmbedDim]),
    );

    x = this.decoderPred.call(
      x,
      getParam('decoder_pred.weight', [this.patchSize * this.patchSize * 3, this.decoderEmbedDim]),
      getParam('decoder_pred.bias', [this.patchSize * this.patchSize * 3]),
    );

    x = recordOp('Slice', [
      x,
      recordOp('Constant', [], { value: [1], dtype: 7 }),
      recordOp('Constant', [], { value: [this.numPatches + 1], dtype: 7 }),
      recordOp('Constant', [], { value: [1], dtype: 7 }),
    ]);
    return x;
  }

  /**
   * Executes `call` on `MaskedAutoencoderViT`.
   * @param x Parameter `x`.
   * @param maskIndices Parameter `maskIndices`.
   * @returns Result of `call`.
   */
  call(x: Tensor, maskIndices: Tensor): Tensor {
    let latent: Tensor;
    [latent, maskIndices] = this.forwardEncoder(x, maskIndices);
    const pred = this.forwardDecoder(latent, maskIndices);
    return pred;
  }
}

/**
 * Executes `maeVitBasePatch16`.
 * @returns Result of `maeVitBasePatch16`.
 */
export function maeVitBasePatch16(): MaskedAutoencoderViT {
  return new MaskedAutoencoderViT(224, 16, 3, 768, 12, 12, 512, 8, 16, 4.0);
}
