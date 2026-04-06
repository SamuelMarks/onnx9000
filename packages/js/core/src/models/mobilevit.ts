import { Tensor } from '../ir/tensor.js';
import {
  BatchNormalization,
  ConvND,
  Gemm,
  LayerNormalization,
  MultiHeadAttention,
  Silu,
} from '../primitives.js';

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

export class MobileViTBlock {
  public prefix: string;
  public dim: number;
  public outDim: number;

  public localConv1: ConvND;
  public localBn1: BatchNormalization;
  public localAct1: Silu;

  public localConv2: ConvND;
  public localBn2: BatchNormalization;
  public localAct2: Silu;

  public attn: MultiHeadAttention;
  public norm1: LayerNormalization;
  public norm2: LayerNormalization;
  public mlpFc1: Gemm;
  public mlpAct: Silu;
  public mlpFc2: Gemm;

  public fusionConv1: ConvND;
  public fusionBn1: BatchNormalization;
  public fusionAct1: Silu;

  public fusionConv2: ConvND;
  public fusionBn2: BatchNormalization;
  public fusionAct2: Silu;

  constructor(dim: number, outDim: number, numHeads: number, mlpDim: number, prefix: string = '') {
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

export class MobileViT {
  public numClasses: number;
  public stemConv: ConvND;
  public stemBn: BatchNormalization;
  public stemAct: Silu;
  public block1: MobileViTBlock;
  public headConv: ConvND;
  public headBn: BatchNormalization;
  public headAct: Silu;
  public classifier: Gemm;

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

export function mobilevitS(numClasses: number = 1000): MobileViT {
  return new MobileViT(numClasses);
}
