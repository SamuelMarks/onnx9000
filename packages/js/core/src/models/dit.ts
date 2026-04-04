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

export class DiTBlock {
  public prefix: string;
  public hiddenSize: number;
  public norm1: LayerNormalization;
  public attn: MultiHeadAttention;
  public norm2: LayerNormalization;
  public mlpFc1: Gemm;
  public act: Gelu;
  public mlpFc2: Gemm;
  public adaLNModulation: Gemm;

  constructor(hiddenSize: number, numHeads: number, mlpRatio: number = 4.0, prefix: string = '') {
    this.prefix = prefix;
    this.hiddenSize = hiddenSize;
    this.norm1 = new LayerNormalization([hiddenSize], 1e-6);
    this.attn = new MultiHeadAttention(numHeads, true, true);
    this.norm2 = new LayerNormalization([hiddenSize], 1e-6);

    this.mlpFc1 = new Gemm(1.0, 1.0, 0, 1);
    this.act = new Gelu();
    this.mlpFc2 = new Gemm(1.0, 1.0, 0, 1);
    this.adaLNModulation = new Gemm(1.0, 1.0, 0, 1);
  }

  call(x: Tensor, c: Tensor): Tensor {
    const cProj = this.adaLNModulation.call(
      c,
      getParam(`${this.prefix}.adaLN_modulation.weight`, [6 * this.hiddenSize, this.hiddenSize]),
      getParam(`${this.prefix}.adaLN_modulation.bias`, [6 * this.hiddenSize]),
    );

    let splitOut = recordOp('Split', [cProj]);

    const axisTensor = recordOp('Constant', [], { value: [1], dtype: 7 });
    let shiftMsa = recordOp('Unsqueeze', [splitOut, axisTensor]);
    let scaleMsa = recordOp('Unsqueeze', [splitOut, axisTensor]);
    let gateMsa = recordOp('Unsqueeze', [splitOut, axisTensor]);
    let shiftMlp = recordOp('Unsqueeze', [splitOut, axisTensor]);
    let scaleMlp = recordOp('Unsqueeze', [splitOut, axisTensor]);
    let gateMlp = recordOp('Unsqueeze', [splitOut, axisTensor]);

    let identity = x;
    let xNorm = this.norm1.call(
      x,
      getParam(`${this.prefix}.norm1.weight`, [this.hiddenSize]),
      getParam(`${this.prefix}.norm1.bias`, [this.hiddenSize]),
    );

    const oneTensor = recordOp('Constant', [], { value: [1.0], dtype: 1 });
    let xModulated = recordOp('Add', [
      recordOp('Mul', [recordOp('Add', [oneTensor, scaleMsa]), xNorm]),
      shiftMsa,
    ]);

    let xAttn = this.attn.call(xModulated, xModulated, xModulated);
    xAttn = recordOp('Mul', [gateMsa, xAttn]);
    x = recordOp('Add', [identity, xAttn]);

    identity = x;
    xNorm = this.norm2.call(
      x,
      getParam(`${this.prefix}.norm2.weight`, [this.hiddenSize]),
      getParam(`${this.prefix}.norm2.bias`, [this.hiddenSize]),
    );

    xModulated = recordOp('Add', [
      recordOp('Mul', [recordOp('Add', [oneTensor, scaleMlp]), xNorm]),
      shiftMlp,
    ]);

    let xMlp = this.mlpFc1.call(
      xModulated,
      getParam(`${this.prefix}.mlp.fc1.weight`, [this.hiddenSize * 4, this.hiddenSize]),
      getParam(`${this.prefix}.mlp.fc1.bias`, [this.hiddenSize * 4]),
    );
    xMlp = this.act.call(xMlp);
    xMlp = this.mlpFc2.call(
      xMlp,
      getParam(`${this.prefix}.mlp.fc2.weight`, [this.hiddenSize, this.hiddenSize * 4]),
      getParam(`${this.prefix}.mlp.fc2.bias`, [this.hiddenSize]),
    );

    xMlp = recordOp('Mul', [gateMlp, xMlp]);
    x = recordOp('Add', [identity, xMlp]);

    return x;
  }
}

export class DiT {
  public hiddenSize: number;
  public outChannels: number;
  public patchEmbed: PatchEmbed;
  public tEmbedder: Gemm;
  public blocks: DiTBlock[];
  public finalLayerNorm: LayerNormalization;
  public finalLayerAdaLN: Gemm;
  public finalLayerProj: Gemm;

  constructor(
    inputSize: number = 32,
    patchSize: number = 2,
    inChannels: number = 4,
    hiddenSize: number = 1152,
    depth: number = 28,
    numHeads: number = 16,
    mlpRatio: number = 4.0,
  ) {
    this.hiddenSize = hiddenSize;
    this.outChannels = inChannels * patchSize * patchSize;
    this.patchEmbed = new PatchEmbed(inputSize, patchSize, inChannels, hiddenSize, 'x_embedder');
    this.tEmbedder = new Gemm(1.0, 1.0, 0, 1);

    this.blocks = [];
    for (let i = 0; i < depth; i++) {
      this.blocks.push(new DiTBlock(hiddenSize, numHeads, mlpRatio, `blocks.${i}`));
    }

    this.finalLayerNorm = new LayerNormalization([hiddenSize], 1e-6);
    this.finalLayerAdaLN = new Gemm(1.0, 1.0, 0, 1);
    this.finalLayerProj = new Gemm(1.0, 1.0, 0, 1);
  }

  call(x: Tensor, t: Tensor): Tensor {
    x = this.patchEmbed.call(x);

    const posEmbed = getParam('pos_embed', [1, this.patchEmbed.numPatches, this.hiddenSize]);
    x = recordOp('Add', [x, posEmbed]);

    const c = this.tEmbedder.call(
      t,
      getParam('t_embedder.weight', [this.hiddenSize, this.hiddenSize]),
      getParam('t_embedder.bias', [this.hiddenSize]),
    );

    for (const block of this.blocks) {
      x = block.call(x, c);
    }

    const cProj = this.finalLayerAdaLN.call(
      c,
      getParam('final_layer.adaLN_modulation.weight', [2 * this.hiddenSize, this.hiddenSize]),
      getParam('final_layer.adaLN_modulation.bias', [2 * this.hiddenSize]),
    );

    let splitOut = recordOp('Split', [cProj]);
    const axisTensor = recordOp('Constant', [], { value: [1], dtype: 7 });
    let shift = recordOp('Unsqueeze', [splitOut, axisTensor]);
    let scale = recordOp('Unsqueeze', [splitOut, axisTensor]);

    x = this.finalLayerNorm.call(
      x,
      getParam('final_layer.norm.weight', [this.hiddenSize]),
      getParam('final_layer.norm.bias', [this.hiddenSize]),
    );
    const oneTensor = recordOp('Constant', [], { value: [1.0], dtype: 1 });
    x = recordOp('Add', [recordOp('Mul', [recordOp('Add', [oneTensor, scale]), x]), shift]);

    x = this.finalLayerProj.call(
      x,
      getParam('final_layer.linear.weight', [this.outChannels, this.hiddenSize]),
      getParam('final_layer.linear.bias', [this.outChannels]),
    );
    return x;
  }
}

export function ditXl2(): DiT {
  return new DiT(32, 2, 4, 1152, 28, 16, 4.0);
}
