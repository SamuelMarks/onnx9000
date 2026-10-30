/**
 * @fileoverview dit.ts
 * Provides dit functionality for the core package.
 */
import { Tensor } from '../ir/tensor.js';
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

/** Represents the `DiTBlock` class. */
export class DiTBlock {
  /** Property `prefix` of `DiTBlock`. */
  public prefix: string;
  /** Property `hiddenSize` of `DiTBlock`. */
  public hiddenSize: number;
  /** Property `norm1` of `DiTBlock`. */
  public norm1: LayerNormalization;
  /** Property `attn` of `DiTBlock`. */
  public attn: MultiHeadAttention;
  /** Property `norm2` of `DiTBlock`. */
  public norm2: LayerNormalization;
  /** Property `mlpFc1` of `DiTBlock`. */
  public mlpFc1: Gemm;
  /** Property `act` of `DiTBlock`. */
  public act: Gelu;
  /** Property `mlpFc2` of `DiTBlock`. */
  public mlpFc2: Gemm;
  /** Property `adaLNModulation` of `DiTBlock`. */
  public adaLNModulation: Gemm;

  /**
   * Initializes a new `DiTBlock` instance.
   * @param hiddenSize Parameter `hiddenSize`.
   * @param numHeads Parameter `numHeads`.
   * @param _mlpRatio Parameter `_mlpRatio`.
   * @param prefix Parameter `prefix`.
   */
  constructor(hiddenSize: number, numHeads: number, _mlpRatio: number = 4.0, prefix: string = '') {
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

  /**
   * Executes `call` on `DiTBlock`.
   * @param x Parameter `x`.
   * @param c Parameter `c`.
   * @returns Result of `call`.
   */
  call(x: Tensor, c: Tensor): Tensor {
    const cProj = this.adaLNModulation.call(
      c,
      getParam(`${this.prefix}.adaLN_modulation.weight`, [6 * this.hiddenSize, this.hiddenSize]),
      getParam(`${this.prefix}.adaLN_modulation.bias`, [6 * this.hiddenSize]),
    );

    const splitOut = recordOp('Split', [cProj]);

    const axisTensor = recordOp('Constant', [], { value: [1], dtype: 7 });
    const shiftMsa = recordOp('Unsqueeze', [splitOut, axisTensor]);
    const scaleMsa = recordOp('Unsqueeze', [splitOut, axisTensor]);
    const gateMsa = recordOp('Unsqueeze', [splitOut, axisTensor]);
    const shiftMlp = recordOp('Unsqueeze', [splitOut, axisTensor]);
    const scaleMlp = recordOp('Unsqueeze', [splitOut, axisTensor]);
    const gateMlp = recordOp('Unsqueeze', [splitOut, axisTensor]);

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

/** Represents the `DiT` class. */
export class DiT {
  /** Property `hiddenSize` of `DiT`. */
  public hiddenSize: number;
  /** Property `outChannels` of `DiT`. */
  public outChannels: number;
  /** Property `patchEmbed` of `DiT`. */
  public patchEmbed: PatchEmbed;
  /** Property `tEmbedder` of `DiT`. */
  public tEmbedder: Gemm;
  /** Property `blocks` of `DiT`. */
  public blocks: DiTBlock[];
  /** Property `finalLayerNorm` of `DiT`. */
  public finalLayerNorm: LayerNormalization;
  /** Property `finalLayerAdaLN` of `DiT`. */
  public finalLayerAdaLN: Gemm;
  /** Property `finalLayerProj` of `DiT`. */
  public finalLayerProj: Gemm;

  /**
   * Initializes a new `DiT` instance.
   * @param inputSize Parameter `inputSize`.
   * @param patchSize Parameter `patchSize`.
   * @param inChannels Parameter `inChannels`.
   * @param hiddenSize Parameter `hiddenSize`.
   * @param depth Parameter `depth`.
   * @param numHeads Parameter `numHeads`.
   * @param mlpRatio Parameter `mlpRatio`.
   */
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

  /**
   * Executes `call` on `DiT`.
   * @param x Parameter `x`.
   * @param t Parameter `t`.
   * @returns Result of `call`.
   */
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

    const splitOut = recordOp('Split', [cProj]);
    const axisTensor = recordOp('Constant', [], { value: [1], dtype: 7 });
    const shift = recordOp('Unsqueeze', [splitOut, axisTensor]);
    const scale = recordOp('Unsqueeze', [splitOut, axisTensor]);

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

/**
 * Executes `ditXl2`.
 * @returns Result of `ditXl2`.
 */
export function ditXl2(): DiT {
  return new DiT(32, 2, 4, 1152, 28, 16, 4.0);
}
