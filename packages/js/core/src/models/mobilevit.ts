/**
 * @fileoverview mobilevit.ts
 * Provides mobilevit functionality for the core package.
 */
import { Tensor } from '../ir/tensor.js';
import { recordOp } from '../macros.js';
import {
  BatchNormalization,
  ConvND,
  Gemm,
  LayerNormalization,
  MultiHeadAttention,
  Silu,
} from '../primitives.js';

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

/** Represents the `MobileViTBlock` class. */
export class MobileViTBlock {
  /** Property `prefix` of `MobileViTBlock`. */
  public prefix: string;
  /** Property `dim` of `MobileViTBlock`. */
  public dim: number;
  /** Property `outDim` of `MobileViTBlock`. */
  public outDim: number;

  /** Property `localConv1` of `MobileViTBlock`. */
  public localConv1: ConvND;
  /** Property `localBn1` of `MobileViTBlock`. */
  public localBn1: BatchNormalization;
  /** Property `localAct1` of `MobileViTBlock`. */
  public localAct1: Silu;

  /** Property `localConv2` of `MobileViTBlock`. */
  public localConv2: ConvND;
  /** Property `localBn2` of `MobileViTBlock`. */
  public localBn2: BatchNormalization;
  /** Property `localAct2` of `MobileViTBlock`. */
  public localAct2: Silu;

  /** Property `attn` of `MobileViTBlock`. */
  public attn: MultiHeadAttention;
  /** Property `norm1` of `MobileViTBlock`. */
  public norm1: LayerNormalization;
  /** Property `norm2` of `MobileViTBlock`. */
  public norm2: LayerNormalization;
  /** Property `mlpFc1` of `MobileViTBlock`. */
  public mlpFc1: Gemm;
  /** Property `mlpAct` of `MobileViTBlock`. */
  public mlpAct: Silu;
  /** Property `mlpFc2` of `MobileViTBlock`. */
  public mlpFc2: Gemm;

  /** Property `fusionConv1` of `MobileViTBlock`. */
  public fusionConv1: ConvND;
  /** Property `fusionBn1` of `MobileViTBlock`. */
  public fusionBn1: BatchNormalization;
  /** Property `fusionAct1` of `MobileViTBlock`. */
  public fusionAct1: Silu;

  /** Property `fusionConv2` of `MobileViTBlock`. */
  public fusionConv2: ConvND;
  /** Property `fusionBn2` of `MobileViTBlock`. */
  public fusionBn2: BatchNormalization;
  /** Property `fusionAct2` of `MobileViTBlock`. */
  public fusionAct2: Silu;

  /**
   * Initializes a new `MobileViTBlock` instance.
   * @param dim Parameter `dim`.
   * @param outDim Parameter `outDim`.
   * @param numHeads Parameter `numHeads`.
   * @param _mlpDim Parameter `_mlpDim`.
   * @param prefix Parameter `prefix`.
   */
  constructor(dim: number, outDim: number, numHeads: number, _mlpDim: number, prefix: string = '') {
    this.prefix = prefix;
    this.dim = dim;
    this.outDim = outDim;

    this.localConv1 = new ConvND(2, dim, dim, 3, 1, 1, 1, 1, false);
    this.localBn1 = new BatchNormalization(dim);
    this.localAct1 = new Silu();

    this.localConv2 = new ConvND(2, dim, dim, 1, 1, 0, 1, 1, false);
    this.localBn2 = new BatchNormalization(dim);
    this.localAct2 = new Silu();

    this.attn = new MultiHeadAttention(numHeads);
    this.norm1 = new LayerNormalization([dim]);
    this.norm2 = new LayerNormalization([dim]);
    this.mlpFc1 = new Gemm(1.0, 1.0, 0, 1);
    this.mlpAct = new Silu();
    this.mlpFc2 = new Gemm(1.0, 1.0, 0, 1);

    this.fusionConv1 = new ConvND(2, dim, dim, 1, 1, 0, 1, 1, false);
    this.fusionBn1 = new BatchNormalization(dim);
    this.fusionAct1 = new Silu();

    this.fusionConv2 = new ConvND(2, dim * 2, outDim, 3, 1, 1, 1, 1, false);
    this.fusionBn2 = new BatchNormalization(outDim);
    this.fusionAct2 = new Silu();
  }

  /**
   * Executes `call` on `MobileViTBlock`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    const identity = x;

    let out = this.localConv1.call(
      x,
      getParam(`${this.prefix}.local_conv1.weight`, [
        this.localConv1.outChannels,
        this.localConv1.inChannels,
        3,
        3,
      ]),
    );
    out = this.localBn1.call(
      out,
      getParam(`${this.prefix}.local_bn1.weight`, [this.dim]),
      getParam(`${this.prefix}.local_bn1.bias`, [this.dim]),
      getParam(`${this.prefix}.local_bn1.running_mean`, [this.dim]),
      getParam(`${this.prefix}.local_bn1.running_var`, [this.dim]),
    );
    out = this.localAct1.call(out);

    out = this.localConv2.call(
      out,
      getParam(`${this.prefix}.local_conv2.weight`, [
        this.localConv2.outChannels,
        this.localConv2.inChannels,
        1,
        1,
      ]),
    );
    out = this.localBn2.call(
      out,
      getParam(`${this.prefix}.local_bn2.weight`, [this.dim]),
      getParam(`${this.prefix}.local_bn2.bias`, [this.dim]),
      getParam(`${this.prefix}.local_bn2.running_mean`, [this.dim]),
      getParam(`${this.prefix}.local_bn2.running_var`, [this.dim]),
    );
    out = this.localAct2.call(out);

    let attnOut = this.norm1.call(
      out,
      getParam(`${this.prefix}.norm1.weight`, [this.dim]),
      getParam(`${this.prefix}.norm1.bias`, [this.dim]),
    );
    attnOut = this.attn.call(attnOut, attnOut, attnOut);
    attnOut = recordOp('Add', [out, attnOut]);

    let mlpOut = this.norm2.call(
      attnOut,
      getParam(`${this.prefix}.norm2.weight`, [this.dim]),
      getParam(`${this.prefix}.norm2.bias`, [this.dim]),
    );
    mlpOut = this.mlpFc1.call(
      mlpOut,
      getParam(`${this.prefix}.mlp_fc1.weight`, [this.dim * 2, this.dim]),
      getParam(`${this.prefix}.mlp_fc1.bias`, [this.dim * 2]),
    );
    mlpOut = this.mlpAct.call(mlpOut);
    mlpOut = this.mlpFc2.call(
      mlpOut,
      getParam(`${this.prefix}.mlp_fc2.weight`, [this.dim, this.dim * 2]),
      getParam(`${this.prefix}.mlp_fc2.bias`, [this.dim]),
    );
    attnOut = recordOp('Add', [attnOut, mlpOut]);

    out = this.fusionConv1.call(
      attnOut,
      getParam(`${this.prefix}.fusion_conv1.weight`, [this.dim, this.dim, 1, 1]),
    );
    out = this.fusionBn1.call(
      out,
      getParam(`${this.prefix}.fusion_bn1.weight`, [this.dim]),
      getParam(`${this.prefix}.fusion_bn1.bias`, [this.dim]),
      getParam(`${this.prefix}.fusion_bn1.running_mean`, [this.dim]),
      getParam(`${this.prefix}.fusion_bn1.running_var`, [this.dim]),
    );
    out = this.fusionAct1.call(out);

    out = recordOp('Concat', [identity, out], { axis: 1 });

    out = this.fusionConv2.call(
      out,
      getParam(`${this.prefix}.fusion_conv2.weight`, [this.outDim, this.dim * 2, 3, 3]),
    );
    out = this.fusionBn2.call(
      out,
      getParam(`${this.prefix}.fusion_bn2.weight`, [this.outDim]),
      getParam(`${this.prefix}.fusion_bn2.bias`, [this.outDim]),
      getParam(`${this.prefix}.fusion_bn2.running_mean`, [this.outDim]),
      getParam(`${this.prefix}.fusion_bn2.running_var`, [this.outDim]),
    );
    out = this.fusionAct2.call(out);

    return out;
  }
}

/** Represents the `MobileViT` class. */
export class MobileViT {
  /** Property `numClasses` of `MobileViT`. */
  public numClasses: number;
  /** Property `stemConv` of `MobileViT`. */
  public stemConv: ConvND;
  /** Property `stemBn` of `MobileViT`. */
  public stemBn: BatchNormalization;
  /** Property `stemAct` of `MobileViT`. */
  public stemAct: Silu;
  /** Property `block1` of `MobileViT`. */
  public block1: MobileViTBlock;
  /** Property `headConv` of `MobileViT`. */
  public headConv: ConvND;
  /** Property `headBn` of `MobileViT`. */
  public headBn: BatchNormalization;
  /** Property `headAct` of `MobileViT`. */
  public headAct: Silu;
  /** Property `classifier` of `MobileViT`. */
  public classifier: Gemm;

  /**
   * Initializes a new `MobileViT` instance.
   * @param numClasses Parameter `numClasses`.
   */
  constructor(numClasses: number = 1000) {
    this.numClasses = numClasses;
    this.stemConv = new ConvND(2, 3, 16, 3, 2, 1, 1, 1, false);
    this.stemBn = new BatchNormalization(16);
    this.stemAct = new Silu();

    this.block1 = new MobileViTBlock(16, 32, 4, 64, 'block1');

    this.headConv = new ConvND(2, 32, 1280, 1, 1, 0, 1, 1, false);
    this.headBn = new BatchNormalization(1280);
    this.headAct = new Silu();

    this.classifier = new Gemm(1.0, 1.0, 0, 1);
  }

  /**
   * Executes `call` on `MobileViT`.
   * @param x Parameter `x`.
   * @returns Result of `call`.
   */
  call(x: Tensor): Tensor {
    x = this.stemConv.call(x, getParam('stem_conv.weight', [16, 3, 3, 3]));
    x = this.stemBn.call(
      x,
      getParam('stem_bn.weight', [16]),
      getParam('stem_bn.bias', [16]),
      getParam('stem_bn.running_mean', [16]),
      getParam('stem_bn.running_var', [16]),
    );
    x = this.stemAct.call(x);

    x = this.block1.call(x);

    x = this.headConv.call(x, getParam('head_conv.weight', [1280, 32, 1, 1]));
    x = this.headBn.call(
      x,
      getParam('head_bn.weight', [1280]),
      getParam('head_bn.bias', [1280]),
      getParam('head_bn.running_mean', [1280]),
      getParam('head_bn.running_var', [1280]),
    );
    x = this.headAct.call(x);

    x = recordOp('GlobalAveragePool', [x]);
    x = recordOp('Flatten', [x]);
    x = this.classifier.call(
      x,
      getParam('classifier.weight', [this.numClasses, 1280]),
      getParam('classifier.bias', [this.numClasses]),
    );

    return x;
  }
}

/**
 * Executes `mobilevitS`.
 * @param numClasses Parameter `numClasses`.
 * @returns Result of `mobilevitS`.
 */
export function mobilevitS(numClasses: number = 1000): MobileViT {
  return new MobileViT(numClasses);
}
