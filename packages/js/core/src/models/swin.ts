/**
 * @fileoverview swin.ts
 * Provides swin functionality for the core package.
 */
import { Tensor } from '../ir/tensor.js';
import { recordOp } from '../macros.js';
import { Gelu, Gemm, LayerNormalization, MultiHeadAttention } from '../primitives.js';
import { PatchEmbed } from './vit.js';

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

/** Represents the `WindowAttention` class. */
export class WindowAttention {
  /** Property `prefix` of `WindowAttention`. */
  public prefix: string;
  /** Property `dim` of `WindowAttention`. */
  public dim: number;
  /** Property `windowSize` of `WindowAttention`. */
  public windowSize: [number, number];
  /** Property `numHeads` of `WindowAttention`. */
  public numHeads: number;
  /** Property `attn` of `WindowAttention`. */
  public attn: MultiHeadAttention;
  /** Property `proj` of `WindowAttention`. */
  public proj: Gemm;

  /**
   * Initializes a new `WindowAttention` instance.
   * @param dim Parameter `dim`.
   * @param windowSize Parameter `windowSize`.
   * @param numHeads Parameter `numHeads`.
   * @param qkvBias Parameter `qkvBias`.
   * @param prefix Parameter `prefix`.
   */
  constructor(
    dim: number,
    windowSize: [number, number],
    numHeads: number,
    qkvBias: boolean = true,
    prefix: string = '',
  ) {
    this.prefix = prefix;
    this.dim = dim;
    this.windowSize = windowSize;
    this.numHeads = numHeads;
    this.attn = new MultiHeadAttention(numHeads, qkvBias, true);
    this.proj = new Gemm(1.0, 1.0, 0, 1);
  }

  /**
   * Executes `call` on `WindowAttention`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    x = this.attn.call(x, x, x);
    x = this.proj.call(
      x,
      getParam(`${this.prefix}.proj.weight`, [this.dim, this.dim]),
      getParam(`${this.prefix}.proj.bias`, [this.dim]),
    );
    return x;
  }
}

/** Represents the `SwinTransformerBlock` class. */
export class SwinTransformerBlock {
  /** Property `prefix` of `SwinTransformerBlock`. */
  public prefix: string;
  /** Property `dim` of `SwinTransformerBlock`. */
  public dim: number;
  /** Property `inputResolution` of `SwinTransformerBlock`. */
  public inputResolution: [number, number];
  /** Property `numHeads` of `SwinTransformerBlock`. */
  public numHeads: number;
  /** Property `windowSize` of `SwinTransformerBlock`. */
  public windowSize: number;
  /** Property `shiftSize` of `SwinTransformerBlock`. */
  public shiftSize: number;
  /** Property `mlpRatio` of `SwinTransformerBlock`. */
  public mlpRatio: number;

  /** Property `norm1` of `SwinTransformerBlock`. */
  public norm1: LayerNormalization;
  /** Property `attn` of `SwinTransformerBlock`. */
  public attn: WindowAttention;
  /** Property `norm2` of `SwinTransformerBlock`. */
  public norm2: LayerNormalization;
  /** Property `mlpFc1` of `SwinTransformerBlock`. */
  public mlpFc1: Gemm;
  /** Property `act` of `SwinTransformerBlock`. */
  public act: Gelu;
  /** Property `mlpFc2` of `SwinTransformerBlock`. */
  public mlpFc2: Gemm;

  /**
   * Initializes a new `SwinTransformerBlock` instance.
   * @param dim Parameter `dim`.
   * @param inputResolution Parameter `inputResolution`.
   * @param numHeads Parameter `numHeads`.
   * @param windowSize Parameter `windowSize`.
   * @param shiftSize Parameter `shiftSize`.
   * @param mlpRatio Parameter `mlpRatio`.
   * @param qkvBias Parameter `qkvBias`.
   * @param prefix Parameter `prefix`.
   */
  constructor(
    dim: number,
    inputResolution: [number, number],
    numHeads: number,
    windowSize: number = 7,
    shiftSize: number = 0,
    mlpRatio: number = 4.0,
    qkvBias: boolean = true,
    prefix: string = '',
  ) {
    this.prefix = prefix;
    this.dim = dim;
    this.inputResolution = inputResolution;
    this.numHeads = numHeads;
    this.windowSize = windowSize;
    this.shiftSize = shiftSize;
    this.mlpRatio = mlpRatio;

    if (Math.min(...this.inputResolution) <= this.windowSize) {
      this.shiftSize = 0;
      this.windowSize = Math.min(...this.inputResolution);
    }

    this.norm1 = new LayerNormalization([dim]);
    this.attn = new WindowAttention(
      dim,
      [this.windowSize, this.windowSize],
      numHeads,
      qkvBias,
      `${prefix}.attn`,
    );
    this.norm2 = new LayerNormalization([dim]);
    this.mlpFc1 = new Gemm(1.0, 1.0, 0, 1);
    this.act = new Gelu();
    this.mlpFc2 = new Gemm(1.0, 1.0, 0, 1);
  }

  /**
   * Executes `call` on `SwinTransformerBlock`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    let identity = x;
    x = this.norm1.call(
      x,
      getParam(`${this.prefix}.norm1.weight`, [this.dim]),
      getParam(`${this.prefix}.norm1.bias`, [this.dim]),
    );

    if (this.shiftSize > 0) {
      x = recordOp('Roll', [x], {
        shifts: [-this.shiftSize, -this.shiftSize],
        axes: [1, 2],
      });
    }

    x = this.attn.call(x);

    if (this.shiftSize > 0) {
      x = recordOp('Roll', [x], {
        shifts: [this.shiftSize, this.shiftSize],
        axes: [1, 2],
      });
    }

    x = recordOp('Add', [x, identity]);

    identity = x;
    x = this.norm2.call(
      x,
      getParam(`${this.prefix}.norm2.weight`, [this.dim]),
      getParam(`${this.prefix}.norm2.bias`, [this.dim]),
    );

    const mlpDim = Math.floor(this.dim * this.mlpRatio);
    x = this.mlpFc1.call(
      x,
      getParam(`${this.prefix}.mlp_fc1.weight`, [mlpDim, this.dim]),
      getParam(`${this.prefix}.mlp_fc1.bias`, [mlpDim]),
    );
    x = this.act.call(x);
    x = this.mlpFc2.call(
      x,
      getParam(`${this.prefix}.mlp_fc2.weight`, [this.dim, mlpDim]),
      getParam(`${this.prefix}.mlp_fc2.bias`, [this.dim]),
    );
    x = recordOp('Add', [x, identity]);

    return x;
  }
}

/** Represents the `SwinTransformer` class. */
export class SwinTransformer {
  /** Property `numClasses` of `SwinTransformer`. */
  public numClasses: number;
  /** Property `embedDim` of `SwinTransformer`. */
  public embedDim: number;
  /** Property `patchEmbed` of `SwinTransformer`. */
  public patchEmbed: PatchEmbed;
  /** Property `layers` of `SwinTransformer`. */
  public layers: SwinTransformerBlock[];
  /** Property `norm` of `SwinTransformer`. */
  public norm: LayerNormalization;
  /** Property `head` of `SwinTransformer`. */
  public head: Gemm;

  /**
   * Initializes a new `SwinTransformer` instance.
   * @param embedDim Parameter `embedDim`.
   * @param depths Parameter `depths`.
   * @param numHeads Parameter `numHeads`.
   * @param windowSize Parameter `windowSize`.
   * @param mlpRatio Parameter `mlpRatio`.
   * @param numClasses Parameter `numClasses`.
   */
  constructor(
    embedDim: number = 96,
    depths: number[] = [2, 2, 6, 2],
    numHeads: number[] = [3, 6, 12, 24],
    windowSize: number = 7,
    mlpRatio: number = 4.0,
    numClasses: number = 1000,
  ) {
    this.numClasses = numClasses;
    this.embedDim = embedDim;
    this.patchEmbed = new PatchEmbed(224, 4, 3, embedDim, 'patch_embed');

    this.layers = [];
    for (let iLayer = 0; iLayer < depths.length; iLayer++) {
      const dim = Math.floor(embedDim * 2 ** iLayer);
      const inputResolution: [number, number] = [
        Math.floor(224 / (4 * 2 ** iLayer)),
        Math.floor(224 / (4 * 2 ** iLayer)),
      ];

      for (let i = 0; i < (depths[iLayer] as number); i++) {
        const shiftSize = i % 2 === 0 ? 0 : Math.floor(windowSize / 2);
        this.layers.push(
          new SwinTransformerBlock(
            dim,
            inputResolution,
            numHeads[iLayer] as number,
            windowSize,
            shiftSize,
            mlpRatio,
            true,
            `layers.${iLayer}.blocks.${i}`,
          ),
        );
      }
    }

    const lastDim = Math.floor(embedDim * 2 ** (depths.length - 1));
    this.norm = new LayerNormalization([lastDim]);
    this.head = new Gemm(1.0, 1.0, 0, 1);
  }

  /**
   * Executes `call` on `SwinTransformer`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    x = this.patchEmbed.call(x);

    for (const block of this.layers) {
      x = block.call(x);
    }

    x = recordOp('ReduceMean', [x], { axes: [1], keepdims: 0 });

    const lastDim = Math.floor(this.embedDim * 2 ** 3);
    x = this.norm.call(x, getParam('norm.weight', [lastDim]), getParam('norm.bias', [lastDim]));
    x = this.head.call(
      x,
      getParam('head.weight', [this.numClasses, lastDim]),
      getParam('head.bias', [this.numClasses]),
    );

    return x;
  }
}

/**
 * Executes `swinT`.
 * @param numClasses Parameter `numClasses`.
 * @returns Result of `swinT`.
 */
export function swinT(numClasses: number = 1000): SwinTransformer {
  return new SwinTransformer(96, [2, 2, 6, 2], [3, 6, 12, 24], 7, 4.0, numClasses);
}
