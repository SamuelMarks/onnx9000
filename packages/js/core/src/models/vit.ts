import { Tensor } from '../ir/tensor.js';
import {
  BatchNormalization,
  ConvND,
  Gemm,
  LayerNormalization,
  MultiHeadAttention,
  Gelu,
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

export class PatchEmbed {
  public prefix: string;
  public imgSize: number;
  public patchSize: number;
  public numPatches: number;
  public proj: ConvND;

  constructor(
    imgSize: number = 224,
    patchSize: number = 16,
    inChans: number = 3,
    embedDim: number = 768,
    prefix: string = '',
  ) {
    this.prefix = prefix;
    this.imgSize = imgSize;
    this.patchSize = patchSize;
    this.numPatches = Math.floor(imgSize / patchSize) * Math.floor(imgSize / patchSize);
    this.proj = new ConvND(2, inChans, embedDim, patchSize, patchSize, 0, 1, 1, false);
  }

  call(x: Tensor): Tensor {
    x = this.proj.call(
      x,
      getParam(`${this.prefix}.proj.weight`, [
        this.proj.outChannels,
        this.proj.inChannels,
        this.patchSize,
        this.patchSize,
      ]),
      getParam(`${this.prefix}.proj.bias`, [this.proj.outChannels]),
    );
    x = recordOp('Flatten', [x], { axis: 2 });
    x = recordOp('Transpose', [x], { perm: [0, 2, 1] });
    return x;
  }
}

export class Block {
  public prefix: string;
  public dim: number;
  public norm1: LayerNormalization;
  public attn: MultiHeadAttention;
  public norm2: LayerNormalization;
  public fc1: Gemm;
  public act: Gelu;
  public fc2: Gemm;
  public mlpHiddenDim: number;

  constructor(
    dim: number,
    numHeads: number,
    mlpRatio: number = 4.0,
    qkvBias: boolean = false,
    prefix: string = '',
  ) {
    this.prefix = prefix;
    this.dim = dim;
    this.norm1 = new LayerNormalization([dim], 1e-6);
    this.attn = new MultiHeadAttention(numHeads, qkvBias, true);
    this.norm2 = new LayerNormalization([dim], 1e-6);
    this.mlpHiddenDim = Math.floor(dim * mlpRatio);
    this.fc1 = new Gemm(1.0, 1.0, 0, 1);
    this.act = new Gelu();
    this.fc2 = new Gemm(1.0, 1.0, 0, 1);
  }

  call(x: Tensor): Tensor {
    let identity = x;
    x = this.norm1.call(
      x,
      getParam(`${this.prefix}.norm1.weight`, [this.dim]),
      getParam(`${this.prefix}.norm1.bias`, [this.dim]),
    );
    x = this.attn.call(x, x, x);
    x = recordOp('Add', [x, identity]);

    identity = x;
    x = this.norm2.call(
      x,
      getParam(`${this.prefix}.norm2.weight`, [this.dim]),
      getParam(`${this.prefix}.norm2.bias`, [this.dim]),
    );
    x = this.fc1.call(
      x,
      getParam(`${this.prefix}.fc1.weight`, [this.mlpHiddenDim, this.dim]),
      getParam(`${this.prefix}.fc1.bias`, [this.mlpHiddenDim]),
    );
    x = this.act.call(x);
    x = this.fc2.call(
      x,
      getParam(`${this.prefix}.fc2.weight`, [this.dim, this.mlpHiddenDim]),
      getParam(`${this.prefix}.fc2.bias`, [this.dim]),
    );
    x = recordOp('Add', [x, identity]);
    return x;
  }
}

export class VisionTransformer {
  public embedDim: number;
  public numClasses: number;
  public patchEmbed: PatchEmbed;
  public blocks: Block[];
  public norm: LayerNormalization;
  public head: Gemm;
  public numPatches: number;

  constructor(
    imgSize: number = 224,
    patchSize: number = 16,
    inChans: number = 3,
    numClasses: number = 1000,
    embedDim: number = 768,
    depth: number = 12,
    numHeads: number = 12,
    mlpRatio: number = 4.0,
    qkvBias: boolean = true,
  ) {
    this.embedDim = embedDim;
    this.numClasses = numClasses;
    this.patchEmbed = new PatchEmbed(imgSize, patchSize, inChans, embedDim, 'patch_embed');

    this.blocks = [];
    for (let i = 0; i < depth; i++) {
      this.blocks.push(new Block(embedDim, numHeads, mlpRatio, qkvBias, `blocks.${i}`));
    }

    this.norm = new LayerNormalization([embedDim], 1e-6);
    this.head = new Gemm(1.0, 1.0, 0, 1);
    this.numPatches = this.patchEmbed.numPatches;
  }

  call(x: Tensor): Tensor {
    x = this.patchEmbed.call(x);

    const clsToken = getParam('cls_token', [1, 1, this.embedDim]);
    x = recordOp('Concat', [clsToken, x], { axis: 1 });

    const posEmbed = getParam('pos_embed', [1, this.numPatches + 1, this.embedDim]);
    x = recordOp('Add', [x, posEmbed]);

    for (const block of this.blocks) {
      x = block.call(x);
    }

    x = this.norm.call(
      x,
      getParam('norm.weight', [this.embedDim]),
      getParam('norm.bias', [this.embedDim]),
    );

    const starts = recordOp('Constant', [], { value: [0], dtype: 7 });
    const ends = recordOp('Constant', [], { value: [1], dtype: 7 });
    const axes = recordOp('Constant', [], { value: [1], dtype: 7 });
    x = recordOp('Slice', [x, starts, ends, axes]);

    x = recordOp('Flatten', [x], { axis: 1 });

    x = this.head.call(
      x,
      getParam('head.weight', [this.numClasses, this.embedDim]),
      getParam('head.bias', [this.numClasses]),
    );
    return x;
  }
}

export function vitBasePatch16_224(numClasses: number = 1000): VisionTransformer {
  return new VisionTransformer(224, 16, 3, numClasses, 768, 12, 12, 4.0, true);
}
