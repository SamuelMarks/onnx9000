import { Tensor } from '../ir/tensor.js';
import { Gelu, Gemm, LayerNormalization, MultiHeadAttention } from '../primitives.js';
import { PatchEmbed } from './vit.js';

function getParam(name: string, shape: number[], dtype: any = 'float32'): Tensor {
  return new Tensor(name, shape, dtype, false, false, new Float32Array());
}

function recordOp(opType: string, inputs: Tensor[], attr?: any): Tensor {
  const dtype = inputs[0]?.dtype ?? 'float32';
  return new Tensor(`${opType}_out`, [], dtype, false, false, new Float32Array());
}

export class WindowAttention {
  public prefix: string;
  public dim: number;
  public windowSize: [number, number];
  public numHeads: number;
  public attn: MultiHeadAttention;
  public proj: Gemm;

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

export class SwinTransformerBlock {
  public prefix: string;
  public dim: number;
  public inputResolution: [number, number];
  public numHeads: number;
  public windowSize: number;
  public shiftSize: number;
  public mlpRatio: number;

  public norm1: LayerNormalization;
  public attn: WindowAttention;
  public norm2: LayerNormalization;
  public mlpFc1: Gemm;
  public act: Gelu;
  public mlpFc2: Gemm;

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

  call(x: Tensor): Tensor {
    let identity = x;
    x = this.norm1.call(
      x,
      getParam(`${this.prefix}.norm1.weight`, [this.dim]),
      getParam(`${this.prefix}.norm1.bias`, [this.dim]),
    );

    if (this.shiftSize > 0) {
      x = recordOp('Roll', [x], { shifts: [-this.shiftSize, -this.shiftSize], axes: [1, 2] });
    }

    x = this.attn.call(x);

    if (this.shiftSize > 0) {
      x = recordOp('Roll', [x], { shifts: [this.shiftSize, this.shiftSize], axes: [1, 2] });
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

export class SwinTransformer {
  public numClasses: number;
  public embedDim: number;
  public patchEmbed: PatchEmbed;
  public layers: SwinTransformerBlock[];
  public norm: LayerNormalization;
  public head: Gemm;

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
      const dim = Math.floor(embedDim * Math.pow(2, iLayer));
      const inputResolution: [number, number] = [
        Math.floor(224 / (4 * Math.pow(2, iLayer))),
        Math.floor(224 / (4 * Math.pow(2, iLayer))),
      ];

      for (let i = 0; i < depths[iLayer]!; i++) {
        const shiftSize = i % 2 === 0 ? 0 : Math.floor(windowSize / 2);
        this.layers.push(
          new SwinTransformerBlock(
            dim,
            inputResolution,
            numHeads[iLayer]!,
            windowSize,
            shiftSize,
            mlpRatio,
            true,
            `layers.${iLayer}.blocks.${i}`,
          ),
        );
      }
    }

    const lastDim = Math.floor(embedDim * Math.pow(2, depths.length - 1));
    this.norm = new LayerNormalization([lastDim]);
    this.head = new Gemm(1.0, 1.0, 0, 1);
  }

  call(x: Tensor): Tensor {
    x = this.patchEmbed.call(x);

    for (const block of this.layers) {
      x = block.call(x);
    }

    x = recordOp('ReduceMean', [x], { axes: [1], keepdims: 0 });

    const lastDim = Math.floor(this.embedDim * Math.pow(2, 3));
    x = this.norm.call(x, getParam('norm.weight', [lastDim]), getParam('norm.bias', [lastDim]));
    x = this.head.call(
      x,
      getParam('head.weight', [this.numClasses, lastDim]),
      getParam('head.bias', [this.numClasses]),
    );

    return x;
  }
}

export function swinT(numClasses: number = 1000): SwinTransformer {
  return new SwinTransformer(96, [2, 2, 6, 2], [3, 6, 12, 24], 7, 4.0, numClasses);
}
